"""Analytics metrics functions for gas pipeline data."""

from __future__ import annotations

import polars as pl


def ramp_risk(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate ramp risk index for pipeline/location entities.

    Ramp risk = p95(|Δ day|) / median(flow)

    Args:
        lf: Lazy frame with columns: pipeline_name, loc_name, eff_gas_day, scheduled_quantity

    Returns:
        Lazy frame with columns: pipeline_name, loc_name, ramp_risk, n_days,
        p95_abs_delta, median_flow
    """
    return (
        lf.sort(["pipeline_name", "loc_name", "eff_gas_day"])
        .with_columns(
            [
                # Calculate daily delta (absolute change from previous day)
                pl.col("scheduled_quantity")
                .diff()
                .abs()
                .over(["pipeline_name", "loc_name"])
                .alias("abs_delta")
            ]
        )
        .group_by(["pipeline_name", "loc_name"])
        .agg(
            [
                pl.len().alias("n_days"),
                pl.col("abs_delta").quantile(0.95).alias("p95_abs_delta"),
                pl.col("scheduled_quantity").median().alias("median_flow"),
            ]
        )
        .with_columns(
            [
                # Calculate ramp risk, handle division by zero and null deltas
                pl.when(pl.col("p95_abs_delta").is_null())
                .then(0.0)  # If no deltas (single day), ramp risk is 0
                .otherwise(
                    pl.col("p95_abs_delta")
                    / pl.max_horizontal([pl.col("median_flow"), pl.lit(1.0)])
                )
                .alias("ramp_risk")
            ]
        )
        .select(
            ["pipeline_name", "loc_name", "ramp_risk", "n_days", "p95_abs_delta", "median_flow"]
        )
    )


def reversal_freq(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate reversal frequency for pipeline connections.

    Reversal frequency = fraction of days with sign change in net flow

    Args:
        lf: Lazy frame with columns: pipeline_name, connecting_entity, eff_gas_day,
            rec_del_sign, scheduled_quantity

    Returns:
        Lazy frame with columns: pipeline_name, connecting_entity, reversal_freq,
        n_days, n_reversals, net_flow_std
    """
    return (
        lf.with_columns(
            [
                # Calculate net flow (positive for receipts, negative for deliveries)
                (pl.col("rec_del_sign") * pl.col("scheduled_quantity")).alias("net_flow")
            ]
        )
        .group_by(["pipeline_name", "connecting_entity", "eff_gas_day"])
        .agg([pl.col("net_flow").sum().alias("daily_net_flow")])
        .sort(["pipeline_name", "connecting_entity", "eff_gas_day"])
        .with_columns(
            [
                # Detect sign changes (reversals)
                (pl.col("daily_net_flow").sign() != pl.col("daily_net_flow").sign().shift(1))
                .over(["pipeline_name", "connecting_entity"])
                .alias("is_reversal")
            ]
        )
        .group_by(["pipeline_name", "connecting_entity"])
        .agg(
            [
                pl.len().alias("n_days"),
                pl.col("is_reversal").sum().alias("n_reversals"),
                pl.col("daily_net_flow").std().alias("net_flow_std"),
            ]
        )
        .with_columns(
            [
                # Calculate reversal frequency
                (pl.col("n_reversals") / pl.max_horizontal([pl.col("n_days"), pl.lit(1)])).alias(
                    "reversal_freq"
                )
            ]
        )
        .select(
            [
                "pipeline_name",
                "connecting_entity",
                "reversal_freq",
                "n_days",
                "n_reversals",
                "net_flow_std",
            ]
        )
    )


def imbalance_pct(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate daily imbalance percentage per pipeline.

    Imbalance% = |receipts - deliveries| / max(receipts, 1) * 100

    Args:
        lf: Lazy frame with columns: pipeline_name, eff_gas_day, rec_del_sign, scheduled_quantity

    Returns:
        Lazy frame with columns: pipeline_name, eff_gas_day, imbalance_pct,
        total_receipts, total_deliveries, net_flow
    """
    return (
        lf.group_by(["pipeline_name", "eff_gas_day"])
        .agg(
            [
                # Sum receipts (positive rec_del_sign)
                pl.when(pl.col("rec_del_sign") > 0)
                .then(pl.col("scheduled_quantity"))
                .otherwise(0)
                .sum()
                .alias("total_receipts"),
                # Sum deliveries (negative rec_del_sign)
                pl.when(pl.col("rec_del_sign") < 0)
                .then(pl.col("scheduled_quantity"))
                .otherwise(0)
                .sum()
                .alias("total_deliveries"),
            ]
        )
        .with_columns(
            [
                # Calculate net flow and imbalance percentage
                (pl.col("total_receipts") - pl.col("total_deliveries")).alias("net_flow"),
                (
                    (pl.col("total_receipts") - pl.col("total_deliveries")).abs()
                    / pl.max_horizontal([pl.col("total_receipts"), pl.lit(1.0)])
                    * 100.0
                ).alias("imbalance_pct"),
            ]
        )
        .select(
            [
                "pipeline_name",
                "eff_gas_day",
                "imbalance_pct",
                "total_receipts",
                "total_deliveries",
                "net_flow",
            ]
        )
        .sort(["pipeline_name", "eff_gas_day"])
    )
