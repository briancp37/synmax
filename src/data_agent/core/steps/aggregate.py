"""Aggregate step implementation."""

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
    """Apply aggregation to a step handle.

    Args:
        handle: Input step handle
        params: Aggregate parameters with format:
            {
                "group": [str],  # columns to group by
                "metric": {
                    "col": str,     # column to aggregate
                    "fn": str       # aggregation function
                },
                "freq": str | None  # optional frequency for time-based grouping
            }
        storage: Handle storage for materialization (optional)
        dataset_digest: Dataset digest for fingerprinting

    Returns:
        New step handle with aggregation applied
    """
    # Load data from handle
    if handle.store == "parquet" and handle.path is not None:
        lf = pl.scan_parquet(handle.path)
    else:
        # For lazy handles, we assume the caller maintains the LazyFrame
        raise ValueError("Aggregate step requires materialized input handle")

    # Extract parameters
    group_cols = params.get("group", [])
    metric = params.get("metric", {})
    freq = params.get("freq")

    # Build aggregation expression
    col = metric.get("col", "scheduled_quantity")
    fn = metric.get("fn", "sum").lower()

    if fn == "sum":
        agg_expr = pl.col(col).sum().alias(f"sum_{col}")
    elif fn == "count":
        agg_expr = pl.len().alias("count")
    elif fn == "avg" or fn == "mean":
        agg_expr = pl.col(col).mean().alias(f"avg_{col}")
    elif fn == "p95":
        agg_expr = pl.col(col).quantile(0.95).alias(f"p95_{col}")
    elif fn == "p50" or fn == "median":
        agg_expr = pl.col(col).median().alias(f"p50_{col}")
    elif fn == "min":
        agg_expr = pl.col(col).min().alias(f"min_{col}")
    elif fn == "max":
        agg_expr = pl.col(col).max().alias(f"max_{col}")
    elif fn == "std":
        agg_expr = pl.col(col).std().alias(f"std_{col}")
    elif fn == "var":
        agg_expr = pl.col(col).var().alias(f"var_{col}")
    else:
        raise ValueError(f"Unsupported aggregation function: {fn}")

    # Apply aggregation
    if group_cols:
        if freq and "eff_gas_day" in group_cols:
            # Use dynamic grouping for time-based aggregation
            other_cols = [c for c in group_cols if c != "eff_gas_day"]
            if other_cols:
                # Group by other columns first, then apply dynamic grouping
                lf = lf.group_by(other_cols + ["eff_gas_day"]).agg(agg_expr)
                lf = lf.group_by_dynamic(
                    index_column="eff_gas_day",
                    every=freq,
                    by=other_cols,
                    closed="left",
                ).agg(pl.col(agg_expr.meta.output_name()).sum())
            else:
                # Just dynamic grouping on date
                lf = lf.group_by_dynamic(
                    index_column="eff_gas_day",
                    every=freq,
                    closed="left",
                ).agg(agg_expr)
        else:
            # Regular groupby
            lf = lf.group_by(group_cols).agg(agg_expr)
    else:
        # No grouping - aggregate entire dataset
        lf = lf.select(agg_expr)

    # Determine if we should materialize
    should_materialize = False
    if storage is not None:
        # Aggregation typically reduces data size significantly
        # Only materialize if explicitly requested or if multiple consumers expected
        should_materialize = params.get("materialize", False)

    if should_materialize and storage is not None:
        # Compute fingerprint for caching
        input_fingerprints = [handle.fingerprint] if handle.fingerprint else []
        fingerprint = storage.compute_fingerprint(
            f"{handle.id}_aggregated",
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
                id=f"{handle.id}_aggregated",
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
                df, fingerprint, f"{handle.id}_aggregated", handle.engine
            )
    else:
        # Return lazy handle
        # Build new schema based on grouping and aggregation
        new_schema = {}
        if handle.schema:
            # Add group columns
            for col in group_cols:
                new_schema[col] = handle.schema.get(col, "Utf8")

            # Add aggregated column
            agg_col_name = agg_expr.meta.output_name()
            if fn in ["sum", "mean", "avg", "p95", "p50", "median", "min", "max", "std", "var"]:
                new_schema[agg_col_name] = "Float64"
            elif fn == "count":
                new_schema[agg_col_name] = "UInt32"

        return create_lazy_handle(
            f"{handle.id}_aggregated",
            schema=new_schema,
            engine=handle.engine,
        )
