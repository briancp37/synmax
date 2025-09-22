"""Data quality rules engine.

This module implements R-001 through R-006 data quality rules as defined in the architecture.
Each rule returns a dictionary with count and samples of violations.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ..config import RULES_CONFIG


def run_rules(
    lf: pl.LazyFrame, pipeline: str | None = None, since: str | None = None
) -> dict[str, dict[str, Any]]:
    """Run all data quality rules and return violation counts and samples.

    Args:
        lf: Lazy frame to analyze
        pipeline: Optional pipeline name filter
        since: Optional date filter (YYYY-MM-DD format)

    Returns:
        Dictionary mapping rule_id to {count: int, samples: list[dict]}
    """
    # Apply optional filters
    filtered_lf = lf
    if pipeline:
        filtered_lf = filtered_lf.filter(pl.col("pipeline_name") == pipeline)
    if since:
        # Convert string date to proper date for comparison
        since_date = pl.lit(since).str.strptime(pl.Date, "%Y-%m-%d")
        filtered_lf = filtered_lf.filter(pl.col("eff_gas_day") >= since_date)

    results = {}

    # Run each rule
    results["R-001"] = _rule_001_missing_geo_on_active(filtered_lf)
    results["R-002"] = _rule_002_duplicate_loc_across_states(filtered_lf)
    results["R-003"] = _rule_003_zero_quantity_streaks(filtered_lf)
    results["R-004"] = _rule_004_pipeline_imbalance(filtered_lf)
    results["R-005"] = _rule_005_schema_mismatch(filtered_lf)
    results["R-006"] = _rule_006_gas_day_gaps(filtered_lf)

    return results


def _rule_001_missing_geo_on_active(lf: pl.LazyFrame) -> dict[str, Any]:
    """R-001: Missing Geo on Active.

    Find records where latitude OR longitude is null while scheduled_quantity > 0.

    Args:
        lf: Lazy frame to analyze

    Returns:
        Dictionary with count and samples
    """
    # Handle the case where geo columns might be missing or of type Null
    schema = lf.collect_schema()
    column_names = schema.names()

    # First, get the base data and apply the filter
    base_filter = pl.col("scheduled_quantity") > 0

    # Check if geo columns exist and add geo null checks based on column types
    if "latitude" not in column_names or "longitude" not in column_names:
        # If geo columns don't exist, all non-null scheduled_quantity records are violations
        violations = lf.filter(base_filter)
    elif schema["latitude"] == pl.Null and schema["longitude"] == pl.Null:
        # If both are null type, all non-null scheduled_quantity records are violations
        violations = lf.filter(base_filter)
    else:
        # Normal case with actual geo data
        geo_null_filter = pl.col("latitude").is_null() | pl.col("longitude").is_null()
        violations = lf.filter(base_filter & geo_null_filter)

    # Select relevant columns for samples (only those that exist)
    select_cols = ["scheduled_quantity"]
    if "eff_gas_day" in column_names:
        select_cols.append("eff_gas_day")
    if "pipeline_name" in column_names:
        select_cols.insert(0, "pipeline_name")
    if "loc_name" in column_names:
        select_cols.insert(-1 if "eff_gas_day" in column_names else len(select_cols), "loc_name")

    violations = violations.select(select_cols)

    # Collect violations for count and samples
    violations_df = violations.collect()
    count = len(violations_df)

    # Get up to 3 samples and add geo info
    samples = []
    for row in violations_df.head(3).iter_rows(named=True):
        sample = dict(row)
        # Add geo info based on column availability and types
        if "latitude" not in column_names:
            sample["latitude"] = None
        elif schema["latitude"] == pl.Null:
            sample["latitude"] = None

        if "longitude" not in column_names:
            sample["longitude"] = None
        elif schema["longitude"] == pl.Null:
            sample["longitude"] = None

        samples.append(sample)

    return {"count": count, "samples": samples}


def _rule_002_duplicate_loc_across_states(lf: pl.LazyFrame) -> dict[str, Any]:
    """R-002: Duplicate loc_name across states.

    Find loc_name values that appear in multiple distinct state_abb values.

    Args:
        lf: Lazy frame to analyze

    Returns:
        Dictionary with count and samples
    """
    schema = lf.collect_schema()
    column_names = schema.names()

    # Skip if required columns don't exist
    if "loc_name" not in column_names or "state_abb" not in column_names:
        return {"count": 0, "samples": []}

    # Group by loc_name and collect unique states
    loc_states = (
        lf.group_by("loc_name")
        .agg(
            [
                pl.col("state_abb").n_unique().alias("n_states"),
                pl.col("state_abb").unique().alias("states"),
                (
                    pl.col("pipeline_name").first().alias("pipeline_name")
                    if "pipeline_name" in column_names
                    else pl.lit(None).alias("pipeline_name")
                ),
                (
                    pl.col("eff_gas_day").first().alias("eff_gas_day")
                    if "eff_gas_day" in column_names
                    else pl.lit(None).alias("eff_gas_day")
                ),
            ]
        )
        .filter(pl.col("n_states") > 1)
    )

    violations_df = loc_states.collect()
    count = len(violations_df)

    # Get up to 3 samples
    samples = []
    for row in violations_df.head(3).iter_rows(named=True):
        samples.append(
            {
                "loc_name": row["loc_name"],
                "n_states": row["n_states"],
                "states": row["states"],
                "pipeline_name": row["pipeline_name"],
                "eff_gas_day": row["eff_gas_day"],
            }
        )

    return {"count": count, "samples": samples}


def _rule_003_zero_quantity_streaks(lf: pl.LazyFrame) -> dict[str, Any]:
    """R-003: Zero-quantity streaks.

    Find locations with scheduled_quantity == 0 for >= N consecutive days
    at locations with nonzero median quantity for the year.

    Args:
        lf: Lazy frame to analyze

    Returns:
        Dictionary with count and samples
    """
    schema = lf.collect_schema()
    column_names = schema.names()

    # Skip if required columns don't exist
    required_cols = ["pipeline_name", "loc_name", "scheduled_quantity", "eff_gas_day"]
    if not all(col in column_names for col in required_cols):
        return {"count": 0, "samples": []}

    streak_days = RULES_CONFIG.get("zero_quantity_streak_days", 7)

    # First, find locations with nonzero median for the year
    active_locations = (
        lf.group_by(["pipeline_name", "loc_name"])
        .agg(pl.col("scheduled_quantity").median().alias("median_quantity"))
        .filter(pl.col("median_quantity") > 0)
        .select(["pipeline_name", "loc_name"])
    )

    # Join back to get only active locations
    active_data = lf.join(active_locations, on=["pipeline_name", "loc_name"], how="inner")

    # For simplicity, we'll identify potential streaks by looking for consecutive zeros
    # This is a simplified implementation - a full implementation would need
    # to properly identify consecutive date ranges
    zero_streaks = (
        active_data.filter(pl.col("scheduled_quantity") == 0)
        .group_by(["pipeline_name", "loc_name"])
        .agg(
            [
                pl.col("eff_gas_day").count().alias("zero_days"),
                pl.col("eff_gas_day").min().alias("first_zero_date"),
                pl.col("eff_gas_day").max().alias("last_zero_date"),
            ]
        )
        .filter(pl.col("zero_days") >= streak_days)
    )

    violations_df = zero_streaks.collect()
    count = len(violations_df)

    # Get up to 3 samples
    samples = []
    for row in violations_df.head(3).iter_rows(named=True):
        samples.append(dict(row))

    return {"count": count, "samples": samples}


def _rule_004_pipeline_imbalance(lf: pl.LazyFrame) -> dict[str, Any]:
    """R-004: Pipeline Imbalance (daily).

    Find daily pipeline imbalances where |sum(receipts) - sum(deliveries)| / max(receipts, 1) >
    threshold.

    Args:
        lf: Lazy frame to analyze

    Returns:
        Dictionary with count and samples
    """
    schema = lf.collect_schema()
    column_names = schema.names()

    # Skip if required columns don't exist
    required_cols = ["pipeline_name", "eff_gas_day", "rec_del_sign", "scheduled_quantity"]
    if not all(col in column_names for col in required_cols):
        return {"count": 0, "samples": []}

    threshold_pct = RULES_CONFIG.get("imbalance_threshold_pct", 5.0) / 100.0

    # Calculate daily pipeline balances
    daily_balance = (
        lf.group_by(["pipeline_name", "eff_gas_day"])
        .agg(
            [
                pl.when(pl.col("rec_del_sign") == 1)
                .then(pl.col("scheduled_quantity"))
                .otherwise(0.0)
                .sum()
                .alias("receipts"),
                pl.when(pl.col("rec_del_sign") == -1)
                .then(pl.col("scheduled_quantity"))
                .otherwise(0.0)
                .sum()
                .alias("deliveries"),
            ]
        )
        .with_columns(
            [
                (pl.col("receipts") - pl.col("deliveries")).alias("net"),
                (pl.col("receipts") - pl.col("deliveries")).abs().alias("imbalance_abs"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("imbalance_abs") / pl.max_horizontal(pl.col("receipts"), pl.lit(1.0))
                ).alias("imbalance_ratio")
            ]
        )
        .filter(pl.col("imbalance_ratio") > threshold_pct)
    )

    violations_df = daily_balance.collect()
    count = len(violations_df)

    # Get up to 3 samples
    samples = []
    for row in violations_df.head(3).iter_rows(named=True):
        samples.append(
            {
                "pipeline_name": row["pipeline_name"],
                "eff_gas_day": row["eff_gas_day"],
                "receipts": row["receipts"],
                "deliveries": row["deliveries"],
                "net": row["net"],
                "imbalance_ratio": row["imbalance_ratio"],
            }
        )

    return {"count": count, "samples": samples}


def _rule_005_schema_mismatch(lf: pl.LazyFrame) -> dict[str, Any]:
    """R-005: Schema mismatch.

    Find records where connecting_pipeline is not null but category_short != 'Interconnect'.

    Args:
        lf: Lazy frame to analyze

    Returns:
        Dictionary with count and samples
    """
    schema = lf.collect_schema()
    column_names = schema.names()

    # Skip if required columns don't exist
    required_cols = ["connecting_pipeline", "category_short"]
    if not all(col in column_names for col in required_cols):
        return {"count": 0, "samples": []}

    # Build select columns based on what's available
    select_cols = ["connecting_pipeline", "category_short"]
    for col in ["pipeline_name", "loc_name", "eff_gas_day"]:
        if col in column_names:
            select_cols.append(col)

    violations = lf.filter(
        pl.col("connecting_pipeline").is_not_null()
        & (pl.col("category_short").str.to_lowercase() != "interconnect")
    ).select(select_cols)

    violations_df = violations.collect()
    count = len(violations_df)

    # Get up to 3 samples
    samples = []
    for row in violations_df.head(3).iter_rows(named=True):
        samples.append(dict(row))

    return {"count": count, "samples": samples}


def _rule_006_gas_day_gaps(lf: pl.LazyFrame) -> dict[str, Any]:
    """R-006: Eff gas day gaps.

    Find missing dates for active nodes over long spans.
    This is a simplified implementation that looks for locations with
    significant gaps in their date coverage.

    Args:
        lf: Lazy frame to analyze

    Returns:
        Dictionary with count and samples
    """
    schema = lf.collect_schema()
    column_names = schema.names()

    # Skip if required columns don't exist
    required_cols = ["pipeline_name", "loc_name", "eff_gas_day", "scheduled_quantity"]
    if not all(col in column_names for col in required_cols):
        return {"count": 0, "samples": []}

    # Find locations with date ranges and count actual vs expected days
    date_coverage = (
        lf.group_by(["pipeline_name", "loc_name"])
        .agg(
            [
                pl.col("eff_gas_day").min().alias("first_date"),
                pl.col("eff_gas_day").max().alias("last_date"),
                pl.col("eff_gas_day").n_unique().alias("actual_days"),
                pl.col("scheduled_quantity").sum().alias("total_quantity"),
            ]
        )
        .filter(pl.col("total_quantity") > 0)  # Only active locations
        .with_columns(
            [
                (pl.col("last_date") - pl.col("first_date"))
                .dt.total_days()
                .cast(pl.Int32)
                .alias("date_span_days")
            ]
        )
        .with_columns(
            [(pl.col("date_span_days") + 1 - pl.col("actual_days")).alias("missing_days")]
        )
        .filter(
            (pl.col("missing_days") > 0)
            & (pl.col("missing_days") / (pl.col("date_span_days") + 1) > 0.1)  # > 10% missing
        )
    )

    violations_df = date_coverage.collect()
    count = len(violations_df)

    # Get up to 3 samples
    samples = []
    for row in violations_df.head(3).iter_rows(named=True):
        samples.append(
            {
                "pipeline_name": row["pipeline_name"],
                "loc_name": row["loc_name"],
                "first_date": row["first_date"],
                "last_date": row["last_date"],
                "actual_days": row["actual_days"],
                "missing_days": row["missing_days"],
                "total_quantity": row["total_quantity"],
            }
        )

    return {"count": count, "samples": samples}
