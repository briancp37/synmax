"""Dataframe operations for query execution."""

from __future__ import annotations

import polars as pl

from .events import changepoint_detection
from .metrics import imbalance_pct, ramp_risk, reversal_freq
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
            # Handle date strings by converting to date literals
            if f.column == "eff_gas_day":
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

                out = out.filter((pl.col(f.column) >= lo_date) & (pl.col(f.column) <= hi_date))
            else:
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

    # Apply analytics operation if specified
    if plan.op == "metric_compute":
        metric_name = plan.op_args.get("name")
        if metric_name == "ramp_risk":
            return ramp_risk(out)
        elif metric_name == "reversal_freq":
            return reversal_freq(out)
        elif metric_name == "imbalance_pct":
            return imbalance_pct(out)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    elif plan.op == "changepoint":
        # Extract parameters for changepoint detection
        groupby_cols = plan.op_args.get("groupby_cols")
        value_col = plan.op_args.get("value_col", "scheduled_quantity")
        date_col = plan.op_args.get("date_col", "eff_gas_day")
        min_size = plan.op_args.get("min_size", 10)
        penalty = plan.op_args.get("penalty", 10.0)
        min_confidence = plan.op_args.get("min_confidence", 0.7)

        # Run changepoint detection and return as lazy frame
        changepoints_df = changepoint_detection(
            out, groupby_cols, value_col, date_col, min_size, penalty, min_confidence
        )
        return changepoints_df.lazy()

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
