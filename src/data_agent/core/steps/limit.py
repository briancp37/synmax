"""Limit step implementation."""

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
    """Apply limit to a step handle.

    Args:
        handle: Input step handle
        params: Limit parameters with format:
            {
                "n": int,           # number of rows to limit to
                "offset": int,      # optional offset (default: 0)
            }
        storage: Handle storage for materialization (optional)
        dataset_digest: Dataset digest for fingerprinting

    Returns:
        New step handle with limit applied
    """
    # Load data from handle
    if handle.store == "parquet" and handle.path is not None:
        lf = pl.scan_parquet(handle.path)
    else:
        # For lazy handles, we assume the caller maintains the LazyFrame
        raise ValueError("Limit step requires materialized input handle")

    # Extract parameters
    n = params.get("n")
    offset = params.get("offset", 0)

    if n is None:
        raise ValueError("Limit step requires 'n' parameter specifying number of rows")

    if n <= 0:
        raise ValueError("Limit 'n' parameter must be positive")

    # Apply limit with optional offset
    if offset > 0:
        lf = lf.slice(offset, n)
    else:
        lf = lf.head(n)

    # Determine if we should materialize
    should_materialize = False
    if storage is not None:
        # Limit typically creates small result sets, materialization usually not needed
        # unless explicitly requested or if this feeds multiple consumers
        should_materialize = params.get("materialize", False)

    if should_materialize and storage is not None:
        # Compute fingerprint for caching
        input_fingerprints = [handle.fingerprint] if handle.fingerprint else []
        fingerprint = storage.compute_fingerprint(
            f"{handle.id}_limited",
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
                id=f"{handle.id}_limited",
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
                df, fingerprint, f"{handle.id}_limited", handle.engine
            )
    else:
        # Return lazy handle
        # Schema stays the same, just fewer rows
        return create_lazy_handle(
            f"{handle.id}_limited",
            schema=handle.schema,
            engine=handle.engine,
        )
