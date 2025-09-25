"""Rank step implementation."""

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
    """Apply ranking to a step handle.

    Args:
        handle: Input step handle
        params: Rank parameters with format:
            {
                "by": str | list[str],      # column(s) to rank by
                "descending": bool,         # sort order (default: True)
                "method": str,              # ranking method (default: "ordinal")
            }
        storage: Handle storage for materialization (optional)
        dataset_digest: Dataset digest for fingerprinting

    Returns:
        New step handle with ranking applied
    """
    # Load data from handle
    if handle.store == "parquet" and handle.path is not None:
        lf = pl.scan_parquet(handle.path)
    else:
        # For lazy handles, we assume the caller maintains the LazyFrame
        raise ValueError("Rank step requires materialized input handle")

    # Extract parameters
    by_cols = params.get("by", [])
    if isinstance(by_cols, str):
        by_cols = [by_cols]

    descending = params.get("descending", True)
    method = params.get("method", "ordinal")

    if not by_cols:
        raise ValueError("Rank step requires 'by' parameter specifying columns to rank by")

    # Apply ranking
    # Create rank expression
    if method == "ordinal":
        rank_expr = pl.int_range(pl.len()).over(pl.col(by_cols).sort(descending=descending)) + 1
    elif method == "dense":
        # Dense ranking - same values get same rank, next rank is consecutive
        rank_expr = pl.col(by_cols[0]).rank(method="dense", descending=descending)
    elif method == "min":
        # Min ranking - same values get same rank, next rank skips
        rank_expr = pl.col(by_cols[0]).rank(method="min", descending=descending)
    elif method == "max":
        # Max ranking - same values get same rank, next rank skips
        rank_expr = pl.col(by_cols[0]).rank(method="max", descending=descending)
    else:
        raise ValueError(f"Unsupported ranking method: {method}")

    # Add rank column and sort by it
    lf = lf.with_columns(rank_expr.alias("rank")).sort("rank")

    # Determine if we should materialize
    should_materialize = False
    if storage is not None:
        # Ranking typically works on small result sets, so materialization may not be needed
        should_materialize = params.get("materialize", False)

    if should_materialize and storage is not None:
        # Compute fingerprint for caching
        input_fingerprints = [handle.fingerprint] if handle.fingerprint else []
        fingerprint = storage.compute_fingerprint(
            f"{handle.id}_ranked",
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
                id=f"{handle.id}_ranked",
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
            return storage.materialize_handle(df, fingerprint, f"{handle.id}_ranked", handle.engine)
    else:
        # Return lazy handle
        new_schema = dict(handle.schema) if handle.schema else {}
        new_schema["rank"] = "UInt32"

        return create_lazy_handle(
            f"{handle.id}_ranked",
            schema=new_schema,
            engine=handle.engine,
        )
