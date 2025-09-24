"""Change-point detection and event card generation."""

from __future__ import annotations

from typing import Any

import polars as pl
import ruptures as rpt  # type: ignore[import-untyped]


def changepoint_detection(
    lf: pl.LazyFrame,
    groupby_cols: list[str] | None = None,
    value_col: str = "scheduled_quantity",
    date_col: str = "eff_gas_day",
    min_size: int = 10,
    penalty: float = 10.0,
) -> pl.DataFrame:
    """Detect change points in time series data using ruptures PELT algorithm.

    Args:
        lf: Input lazy frame with time series data
        groupby_cols: Optional columns to group by (e.g., pipeline_name)
        value_col: Column containing values to analyze for change points
        date_col: Column containing dates
        min_size: Minimum segment size for change point detection
        penalty: Penalty parameter for PELT algorithm (higher = fewer change points)

    Returns:
        DataFrame with change point events and statistics
    """
    # Collect the data for change point analysis
    if groupby_cols:
        # Group by specified columns and aggregate by date
        df = (
            lf.group_by(groupby_cols + [date_col])
            .agg(pl.col(value_col).sum().alias("total_value"))
            .sort(groupby_cols + [date_col])
            .collect()
        )
    else:
        # Just aggregate by date
        df = (
            lf.group_by(date_col)
            .agg(pl.col(value_col).sum().alias("total_value"))
            .sort(date_col)
            .collect()
        )
        groupby_cols = []

    events = []

    if groupby_cols:
        # Process each group separately
        for group_values in df.select(groupby_cols).unique().iter_rows():
            # Filter data for this group
            group_filter = True
            for i, col in enumerate(groupby_cols):
                group_filter = group_filter & (pl.col(col) == group_values[i])

            group_df = df.filter(group_filter).sort(date_col)

            if len(group_df) < min_size * 2:
                continue  # Skip groups with insufficient data

            # Extract time series values
            values = group_df["total_value"].to_numpy()
            dates = group_df[date_col].to_list()

            # Detect change points
            group_events = _detect_changepoints_for_series(
                values, dates, group_values, groupby_cols, min_size, penalty
            )
            events.extend(group_events)
    else:
        # Process single time series
        if len(df) < min_size * 2:
            return pl.DataFrame(
                schema={
                    "changepoint_date": pl.Date,
                    "before_mean": pl.Float64,
                    "after_mean": pl.Float64,
                    "before_std": pl.Float64,
                    "after_std": pl.Float64,
                    "change_magnitude": pl.Float64,
                    "confidence": pl.Float64,
                }
            )

        values = df["total_value"].to_numpy()
        dates = df[date_col].to_list()

        events = _detect_changepoints_for_series(values, dates, (), [], min_size, penalty)

    if not events:
        # Return empty DataFrame with correct schema
        schema = {
            "changepoint_date": pl.Date,
            "before_mean": pl.Float64,
            "after_mean": pl.Float64,
            "before_std": pl.Float64,
            "after_std": pl.Float64,
            "change_magnitude": pl.Float64,
            "confidence": pl.Float64,
        }
        # Add groupby columns to schema
        for col in groupby_cols:
            schema[col] = pl.Utf8

        return pl.DataFrame(schema=schema)

    return pl.DataFrame(events)


def _detect_changepoints_for_series(
    values: Any,
    dates: list[Any],
    group_values: tuple[Any, ...],
    groupby_cols: list[str],
    min_size: int,
    penalty: float,
) -> list[dict[str, Any]]:
    """Detect change points for a single time series."""
    import numpy as np

    # Use PELT algorithm with L2 cost
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(values)
    changepoints = algo.predict(pen=penalty)

    # Remove the last point (it's always the end of the series)
    if changepoints and changepoints[-1] == len(values):
        changepoints = changepoints[:-1]

    events = []
    for cp_idx in changepoints:
        if cp_idx == 0 or cp_idx >= len(values):
            continue

        # Calculate before/after statistics
        before_values = values[:cp_idx]
        after_values = values[cp_idx:]

        if len(before_values) == 0 or len(after_values) == 0:
            continue

        before_mean = float(np.mean(before_values))
        after_mean = float(np.mean(after_values))
        before_std = float(np.std(before_values))
        after_std = float(np.std(after_values))

        # Calculate change magnitude as relative change
        change_magnitude = float(abs(after_mean - before_mean) / (abs(before_mean) + 1e-8))

        # Simple confidence metric based on effect size
        pooled_std = np.sqrt((before_std**2 + after_std**2) / 2)
        confidence = float(abs(after_mean - before_mean) / (pooled_std + 1e-8))

        event = {
            "changepoint_date": dates[cp_idx],
            "before_mean": before_mean,
            "after_mean": after_mean,
            "before_std": before_std,
            "after_std": after_std,
            "change_magnitude": change_magnitude,
            "confidence": confidence,
        }

        # Add group information
        for i, col in enumerate(groupby_cols):
            event[col] = group_values[i]

        events.append(event)

    return events


def build_event_card(
    changepoints_df: pl.DataFrame,
    original_lf: pl.LazyFrame,
    groupby_cols: list[str] | None = None,
    date_col: str = "eff_gas_day",
    top_n_contributors: int = 5,
) -> dict[str, Any]:
    """Build an event card with before/after stats and top contributors.

    Args:
        changepoints_df: DataFrame with detected change points
        original_lf: Original lazy frame for contributor analysis
        groupby_cols: Groupby columns used in change point detection
        date_col: Date column name
        top_n_contributors: Number of top contributors to include

    Returns:
        Dictionary containing event card information
    """
    if changepoints_df.is_empty():
        return {
            "events_detected": 0,
            "changepoints": [],
            "summary": "No significant change points detected in the time series.",
        }

    events = []
    for row in changepoints_df.iter_rows(named=True):
        changepoint_date = row["changepoint_date"]

        # Find top contributors by analyzing differences around the change point
        contributors = _analyze_contributors(
            original_lf, changepoint_date, groupby_cols, date_col, top_n_contributors
        )

        event_info = {
            "date": str(changepoint_date),
            "before_mean": row["before_mean"],
            "after_mean": row["after_mean"],
            "before_std": row["before_std"],
            "after_std": row["after_std"],
            "change_magnitude": row["change_magnitude"],
            "confidence": row["confidence"],
            "top_contributors": contributors,
        }

        # Add group information if present
        if groupby_cols:
            event_info["group"] = {col: row[col] for col in groupby_cols if col in row}

        events.append(event_info)

    return {
        "events_detected": len(events),
        "changepoints": events,
        "summary": f"Detected {len(events)} significant change point(s) in the time series.",
    }


def _analyze_contributors(
    lf: pl.LazyFrame,
    changepoint_date: Any,
    groupby_cols: list[str] | None,
    date_col: str,
    top_n: int,
) -> list[dict[str, Any]]:
    """Analyze top contributors to a change point by comparing before/after periods."""
    # Define before and after periods (e.g., 30 days before/after)
    import polars as pl

    # Convert changepoint_date to proper date if it's a string
    if isinstance(changepoint_date, str):
        cp_date = pl.lit(changepoint_date).str.to_date()
    else:
        cp_date = pl.lit(changepoint_date)

    # Calculate 30 days before and after
    before_start = cp_date - pl.duration(days=30)
    before_end = cp_date - pl.duration(days=1)
    after_start = cp_date
    after_end = cp_date + pl.duration(days=30)

    # Get before period aggregation
    before_df = (
        lf.filter((pl.col(date_col) >= before_start) & (pl.col(date_col) <= before_end))
        .group_by(["pipeline_name", "loc_name"])
        .agg(pl.col("scheduled_quantity").sum().alias("before_total"))
        .collect()
    )

    # Get after period aggregation
    after_df = (
        lf.filter((pl.col(date_col) >= after_start) & (pl.col(date_col) <= after_end))
        .group_by(["pipeline_name", "loc_name"])
        .agg(pl.col("scheduled_quantity").sum().alias("after_total"))
        .collect()
    )

    # Join and calculate differences
    diff_df = (
        before_df.join(after_df, on=["pipeline_name", "loc_name"], how="full")
        .with_columns([pl.col("before_total").fill_null(0), pl.col("after_total").fill_null(0)])
        .with_columns((pl.col("after_total") - pl.col("before_total")).alias("change"))
        .with_columns(pl.col("change").abs().alias("abs_change"))
        .sort("abs_change", descending=True)
        .head(top_n)
    )

    contributors = []
    for row in diff_df.iter_rows(named=True):
        contributors.append(
            {
                "pipeline_name": row["pipeline_name"],
                "loc_name": row["loc_name"],
                "before_total": row["before_total"],
                "after_total": row["after_total"],
                "change": row["change"],
                "abs_change": row["abs_change"],
            }
        )

    return contributors
