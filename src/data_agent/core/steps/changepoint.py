"""Changepoint detection step implementation using PELT algorithm."""

from __future__ import annotations

from typing import Any

import polars as pl

from ..events import changepoint_detection
from ..handles import HandleStorage, StepHandle, StepStats, create_lazy_handle


def run(
    handle: StepHandle,
    params: dict[str, Any],
    storage: HandleStorage | None = None,
    dataset_digest: str = "",
) -> StepHandle:
    """Apply changepoint detection using PELT (L2) algorithm.

    Args:
        handle: Input step handle
        params: Changepoint parameters with format:
            {
                "column": str,              # column to analyze for changepoints
                "date_column": str,         # date column (default: "eff_gas_day")
                "method": str,              # detection method (default: "pelt")
                "cost": str,                # cost function (default: "l2")
                "min_size": int,            # minimum segment size (default: 7)
                "penalty": float,           # PELT penalty parameter (default: 10.0)
                "min_confidence": float,    # minimum confidence threshold (default: 0.7)
                "groupby": list[str]        # optional grouping columns
            }
        storage: Handle storage for materialization (optional)
        dataset_digest: Dataset digest for fingerprinting

    Returns:
        New step handle with changepoint detection results
    """
    # Load data from handle
    if handle.store == "parquet" and handle.path is not None:
        lf = pl.scan_parquet(handle.path)
    else:
        # For lazy handles, we assume the caller maintains the LazyFrame
        raise ValueError("Changepoint step requires materialized input handle")

    # Extract parameters
    column = params.get("column", "scheduled_quantity")
    date_column = params.get("date_column", "eff_gas_day")
    method = params.get("method", "pelt")
    cost = params.get("cost", "l2")
    min_size = params.get("min_size", 7)
    penalty = params.get("penalty", 10.0)
    min_confidence = params.get("min_confidence", 0.7)
    groupby_cols = params.get("groupby")

    # Validate method and cost
    if method != "pelt":
        raise ValueError(f"Unsupported changepoint method: {method}. Only 'pelt' is supported.")
    if cost != "l2":
        raise ValueError(f"Unsupported cost function: {cost}. Only 'l2' is supported.")

    # Apply changepoint detection using existing implementation
    changepoints_df = changepoint_detection(
        lf=lf,
        groupby_cols=groupby_cols,
        value_col=column,
        date_col=date_column,
        min_size=min_size,
        penalty=penalty,
        min_confidence=min_confidence,
    )

    # Determine if we should materialize
    should_materialize = True  # Changepoint detection is expensive, always materialize
    if storage is not None and should_materialize:
        # Compute fingerprint for caching
        input_fingerprints = [handle.fingerprint] if handle.fingerprint else []
        fingerprint = storage.compute_fingerprint(
            f"{handle.id}_changepoints",
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
                id=f"{handle.id}_changepoints",
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
                changepoints_df, fingerprint, f"{handle.id}_changepoints", handle.engine
            )
    else:
        # Return lazy handle
        # Schema for changepoint results
        new_schema = {
            "changepoint_date": "Date",
            "before_mean": "Float64",
            "after_mean": "Float64",
            "before_std": "Float64",
            "after_std": "Float64",
            "change_magnitude": "Float64",
            "confidence": "Float64",
        }

        # Add groupby columns to schema if present
        if groupby_cols and handle.schema:
            for col in groupby_cols:
                new_schema[col] = handle.schema.get(col, "Utf8")

        return create_lazy_handle(
            f"{handle.id}_changepoints",
            schema=new_schema,
            engine=handle.engine,
        )
