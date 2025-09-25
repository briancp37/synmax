"""Evidence collect step implementation."""

from __future__ import annotations

import time
from typing import Any

import polars as pl

from ..handles import HandleStorage, StepHandle, StepStats, create_lazy_handle


def run(
    handle: StepHandle,
    params: dict[str, Any],
    storage: HandleStorage | None = None,
    dataset_digest: str = "",
) -> StepHandle:
    """Collect evidence and summary metrics from a step handle.

    Args:
        handle: Input step handle
        params: Evidence collection parameters with format:
            {
                "metrics": list[str],       # metrics to collect (optional)
                "include_schema": bool,     # include schema info (default: True)
                "include_stats": bool,      # include data statistics (default: True)
                "sample_size": int,         # number of sample rows (default: 5)
            }
        storage: Handle storage for materialization (optional)
        dataset_digest: Dataset digest for fingerprinting

    Returns:
        New step handle with evidence data
    """
    # Load data from handle
    if handle.store == "parquet" and handle.path is not None:
        lf = pl.scan_parquet(handle.path)
    else:
        # For lazy handles, we assume the caller maintains the LazyFrame
        raise ValueError("Evidence collect step requires materialized input handle")

    # Extract parameters
    requested_metrics = params.get("metrics", [])
    include_schema = params.get("include_schema", True)
    include_stats = params.get("include_stats", True)
    sample_size = params.get("sample_size", 5)

    # Collect data for analysis
    df = lf.collect()

    # Build evidence data
    evidence_data = {
        "source_handle_id": handle.id,
        "source_fingerprint": handle.fingerprint,
        "collected_at": time.time(),
        "dataset_digest": dataset_digest,
    }

    # Basic metrics
    evidence_data["basic_metrics"] = {
        "row_count": df.height,
        "column_count": df.width,
        "memory_usage_bytes": df.estimated_size(),
    }

    # Schema information
    if include_schema:
        evidence_data["schema"] = {
            "columns": df.columns,
            "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        }

    # Data statistics
    if include_stats:
        stats = {}

        # Null counts for all columns
        null_counts = {col: df[col].null_count() for col in df.columns}
        stats["null_counts"] = null_counts
        stats["null_percentages"] = {
            col: (count / df.height * 100) if df.height > 0 else 0.0
            for col, count in null_counts.items()
        }

        # Statistics for numeric columns
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype.is_numeric()]
        if numeric_cols:
            numeric_stats = {}
            for col in numeric_cols:
                series = df[col]
                numeric_stats[col] = {
                    "mean": float(series.mean()) if series.mean() is not None else None,
                    "std": float(series.std()) if series.std() is not None else None,
                    "min": float(series.min()) if series.min() is not None else None,
                    "max": float(series.max()) if series.max() is not None else None,
                    "median": float(series.median()) if series.median() is not None else None,
                    "q25": (
                        float(series.quantile(0.25)) if series.quantile(0.25) is not None else None
                    ),
                    "q75": (
                        float(series.quantile(0.75)) if series.quantile(0.75) is not None else None
                    ),
                    "unique_count": series.n_unique(),
                }
            stats["numeric_statistics"] = numeric_stats

        # Statistics for string columns
        string_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
        if string_cols:
            string_stats = {}
            for col in string_cols:
                series = df[col]
                string_stats[col] = {
                    "unique_count": series.n_unique(),
                    "most_common": (
                        series.value_counts().head(3).to_dicts() if not series.is_empty() else []
                    ),
                    "avg_length": (
                        float(series.str.len_chars().mean())
                        if series.str.len_chars().mean() is not None
                        else None
                    ),
                }
            stats["string_statistics"] = string_stats

        evidence_data["statistics"] = stats

    # Sample data
    if sample_size > 0 and df.height > 0:
        sample_df = df.head(min(sample_size, df.height))
        evidence_data["sample_data"] = sample_df.to_dicts()

    # Custom metrics if requested
    if requested_metrics:
        custom_metrics = {}
        for metric in requested_metrics:
            try:
                if metric == "data_quality_score":
                    # Simple data quality score based on null percentages
                    null_pct = sum(evidence_data["statistics"]["null_percentages"].values())
                    avg_null_pct = null_pct / len(df.columns) if len(df.columns) > 0 else 0
                    custom_metrics[metric] = max(0, 100 - avg_null_pct)

                elif metric == "completeness":
                    # Data completeness percentage
                    total_cells = df.height * df.width
                    null_cells = sum(evidence_data["statistics"]["null_counts"].values())
                    custom_metrics[metric] = (
                        ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0
                    )

                elif metric == "uniqueness":
                    # Average uniqueness across columns
                    uniqueness_scores = []
                    for col in df.columns:
                        unique_count = df[col].n_unique()
                        uniqueness_scores.append(unique_count / df.height if df.height > 0 else 0)
                    custom_metrics[metric] = (
                        sum(uniqueness_scores) / len(uniqueness_scores) * 100
                        if uniqueness_scores
                        else 0
                    )

                else:
                    custom_metrics[metric] = f"Unknown metric: {metric}"

            except Exception as e:
                custom_metrics[metric] = f"Error computing metric: {str(e)}"

        evidence_data["custom_metrics"] = custom_metrics

    # Convert evidence to DataFrame
    evidence_df = pl.DataFrame([evidence_data])

    # Determine if we should materialize
    should_materialize = True  # Evidence collection always materializes
    if storage is not None and should_materialize:
        # Compute fingerprint for caching
        input_fingerprints = [handle.fingerprint] if handle.fingerprint else []
        fingerprint = storage.compute_fingerprint(
            f"{handle.id}_evidence",
            params,
            input_fingerprints,
            dataset_digest,
        )

        # Check if already cached
        cached_path = storage.get_storage_path(fingerprint)
        if cached_path.exists():
            # Load from cache
            df_cached = pl.read_parquet(cached_path)
            stats = StepStats(
                rows=df_cached.height,
                bytes=cached_path.stat().st_size,
                columns=df_cached.width,
                null_count={col: df_cached[col].null_count() for col in df_cached.columns},
                computed_at=cached_path.stat().st_mtime,
            )
            schema = {col: str(dtype) for col, dtype in zip(df_cached.columns, df_cached.dtypes)}

            return StepHandle(
                id=f"{handle.id}_evidence",
                store="parquet",
                path=cached_path,
                engine=handle.engine,
                schema=schema,
                stats=stats,
                fingerprint=fingerprint,
            )
        else:
            # Materialize and cache
            return storage.materialize_handle(
                evidence_df, fingerprint, f"{handle.id}_evidence", handle.engine
            )
    else:
        # Return lazy handle
        schema = {col: str(dtype) for col, dtype in zip(evidence_df.columns, evidence_df.dtypes)}

        return create_lazy_handle(
            f"{handle.id}_evidence",
            schema=schema,
            engine=handle.engine,
        )
