"""Dataframe operations for query execution."""

from __future__ import annotations

import polars as pl

from .plan_schema import Plan


def apply_plan(lf: pl.LazyFrame, plan: Plan) -> pl.LazyFrame:
    """Apply a query plan to a lazy frame.

    Args:
        lf: Input lazy frame
        plan: Query plan to apply

    Returns:
        Transformed lazy frame
    """
    out = lf

    # Apply filters
    for f in plan.filters:
        if f.op == "=":
            out = out.filter(pl.col(f.column) == f.value)
        elif f.op == "between":
            lo, hi = f.value
            out = out.filter((pl.col(f.column) >= lo) & (pl.col(f.column) <= hi))
        elif f.op == "in":
            out = out.filter(pl.col(f.column).is_in(f.value))
        elif f.op == "is_not_null":
            out = out.filter(pl.col(f.column).is_not_null())
        elif f.op == "contains":
            out = out.filter(pl.col(f.column).cast(pl.Utf8).str.contains(str(f.value)))

    # Apply resampling if specified
    if plan.resample:
        # For now, resampling is handled by using daily rollups
        # This would be implemented with groupby_dynamic in a full implementation
        pass

    # Apply aggregation
    if plan.aggregate:
        gb = plan.aggregate.groupby
        aggs = []
        for m in plan.aggregate.metrics:
            col, fn = m["col"], m["fn"].lower()
            if fn == "sum":
                aggs.append(pl.col(col).sum().alias(f"sum_{col}"))
            elif fn == "count":
                aggs.append(pl.len().alias("count"))
            elif fn == "avg":
                aggs.append(pl.col(col).mean().alias(f"avg_{col}"))
            elif fn == "p95":
                aggs.append(pl.col(col).quantile(0.95).alias(f"p95_{col}"))
            elif fn == "p50":
                aggs.append(pl.col(col).median().alias(f"p50_{col}"))

        if gb:
            out = out.group_by(gb).agg(aggs)
        else:
            out = out.select(aggs)

    # Sort/limit handled by executor after collect
    return out
