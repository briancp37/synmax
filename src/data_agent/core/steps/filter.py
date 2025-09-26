"""Filter step implementation."""

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
    """Apply filters to a step handle.

    Args:
        handle: Input step handle
        params: Filter parameters with format:
            {
                "filters": [
                    {"column": str, "op": "="|"in"|"between"|"is_not_null"|"contains", "value": Any}
                ]
            }
        storage: Handle storage for materialization (optional)
        dataset_digest: Dataset digest for fingerprinting

    Returns:
        New step handle with filters applied
    """
    # Load data from handle
    if handle.store == "parquet" and handle.path is not None:
        lf = pl.scan_parquet(handle.path)
    else:
        # For lazy handles, we assume the caller maintains the LazyFrame
        # In practice, this would be handled by the executor
        raise ValueError("Filter step requires materialized input handle")

    # Apply filters from params
    filters = params.get("filters", [])

    for f in filters:
        column = f["column"]
        op = f["op"]
        value = f["value"]

        if op == "=":
            lf = lf.filter(pl.col(column) == value)
        elif op == "between":
            lo, hi = value
            # Handle date strings by converting to date literals
            if column == "eff_gas_day":
                # Handle different date input types
                if isinstance(lo, str):
                    lo_date = pl.lit(lo).str.to_date()
                elif hasattr(lo, "_pyexpr"):  # Polars expression
                    lo_date = lo
                else:
                    lo_date = pl.lit(lo)

                if isinstance(hi, str):
                    hi_date = pl.lit(hi).str.to_date()
                elif hasattr(hi, "_pyexpr"):  # Polars expression
                    hi_date = hi
                else:
                    hi_date = pl.lit(hi)

                lf = lf.filter((pl.col(column) >= lo_date) & (pl.col(column) <= hi_date))
            else:
                lf = lf.filter((pl.col(column) >= lo) & (pl.col(column) <= hi))
        elif op == "in":
            lf = lf.filter(pl.col(column).is_in(value))
        elif op == "is_not_null":
            lf = lf.filter(pl.col(column).is_not_null())
        elif op == "contains":
            lf = lf.filter(pl.col(column).cast(pl.Utf8).str.contains(str(value)))

    # Determine if we should materialize
    should_materialize = False
    if storage is not None:
        # For now, keep filters lazy unless explicitly requested
        # In a full implementation, this would use materialization heuristics
        should_materialize = params.get("materialize", False)

    if should_materialize and storage is not None:
        # Compute fingerprint for caching
        input_fingerprints = [handle.fingerprint] if handle.fingerprint else []
        fingerprint = storage.compute_fingerprint(
            f"{handle.id}_filtered",
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
                id=f"{handle.id}_filtered",
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
                df, fingerprint, f"{handle.id}_filtered", handle.engine
            )
    else:
        # Return lazy handle
        # Note: In practice, the executor would maintain the LazyFrame reference
        schema = (
            handle.schema or {}
        )  # Preserve input schema (filtering doesn't change column types)
        return create_lazy_handle(
            f"{handle.id}_filtered",
            schema=schema,
            engine=handle.engine,
        )
