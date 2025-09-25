"""Resample step implementation."""

from __future__ import annotations

from typing import Any

import polars as pl

from ..handles import HandleStorage, StepHandle, StepStats, create_lazy_handle


def run(
    handle: StepHandle,
    params: dict[str, Any],
    storage: HandleStorage | None = None,
    dataset_digest: str = "",
) -> StepHandle:
    """Apply resampling to a step handle.

    Args:
        handle: Input step handle
        params: Resample parameters with format:
            {
                "freq": str,  # e.g., "1d", "1h", "1w"
                "on": str,    # date column to resample on
                "agg": dict   # aggregation specification
            }
        storage: Handle storage for materialization (optional)
        dataset_digest: Dataset digest for fingerprinting

    Returns:
        New step handle with resampling applied
    """
    # Load data from handle
    if handle.store == "parquet" and handle.path is not None:
        lf = pl.scan_parquet(handle.path)
    else:
        # For lazy handles, we assume the caller maintains the LazyFrame
        raise ValueError("Resample step requires materialized input handle")

    # Extract parameters
    freq = params.get("freq", "1d")
    on_col = params.get("on", "eff_gas_day")
    agg_spec = params.get("agg", {"scheduled_quantity": "sum"})

    # Build aggregation expressions
    agg_exprs = []
    for col, func in agg_spec.items():
        if func == "sum":
            agg_exprs.append(pl.col(col).sum().alias(f"{col}"))
        elif func == "mean":
            agg_exprs.append(pl.col(col).mean().alias(f"{col}"))
        elif func == "count":
            agg_exprs.append(pl.len().alias(f"{col}_count"))
        elif func == "min":
            agg_exprs.append(pl.col(col).min().alias(f"{col}"))
        elif func == "max":
            agg_exprs.append(pl.col(col).max().alias(f"{col}"))
        elif func == "first":
            agg_exprs.append(pl.col(col).first().alias(f"{col}"))
        elif func == "last":
            agg_exprs.append(pl.col(col).last().alias(f"{col}"))

    # Apply resampling using group_by_dynamic
    try:
        lf = lf.group_by_dynamic(
            index_column=on_col,
            every=freq,
            closed="left",
        ).agg(agg_exprs)
    except Exception:
        # Fallback to regular groupby if dynamic groupby fails
        # This can happen if the date column is not properly formatted
        lf = lf.group_by(on_col).agg(agg_exprs)

    # Determine if we should materialize
    should_materialize = False
    if storage is not None:
        # Resampling often reduces data size, so materialization may not be needed
        # unless explicitly requested or if this is an expensive operation
        should_materialize = params.get("materialize", False)

    if should_materialize and storage is not None:
        # Compute fingerprint for caching
        input_fingerprints = [handle.fingerprint] if handle.fingerprint else []
        fingerprint = storage.compute_fingerprint(
            f"{handle.id}_resampled",
            params,
            input_fingerprints,
            dataset_digest,
        )

        # Check if already cached
        cached_path = storage.get_storage_path(fingerprint)
        if cached_path.exists():
            # Load from cache
            df = pl.read_parquet(cached_path)
            stats = StepStats(
                rows=df.height,
                bytes=cached_path.stat().st_size,
                columns=df.width,
                null_count={col: df[col].null_count() for col in df.columns},
                computed_at=cached_path.stat().st_mtime,
            )
            schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

            return StepHandle(
                id=f"{handle.id}_resampled",
                store="parquet",
                path=cached_path,
                engine=handle.engine,
                schema=schema,
                stats=stats,
                fingerprint=fingerprint,
            )
        else:
            # Materialize and cache
            df = lf.collect()
            return storage.materialize_handle(
                df, fingerprint, f"{handle.id}_resampled", handle.engine
            )
    else:
        # Return lazy handle
        # Schema will be modified by resampling - estimate new schema
        new_schema = {}
        if handle.schema:
            # Keep the date column
            new_schema[on_col] = handle.schema.get(on_col, "Date")
            # Add aggregated columns
            for col, func in agg_spec.items():
                if func in ["sum", "mean", "min", "max"]:
                    new_schema[col] = handle.schema.get(col, "Float64")
                elif func == "count":
                    new_schema[f"{col}_count"] = "UInt32"
                elif func in ["first", "last"]:
                    new_schema[col] = handle.schema.get(col, "Utf8")

        return create_lazy_handle(
            f"{handle.id}_resampled",
            schema=new_schema,
            engine=handle.engine,
        )
