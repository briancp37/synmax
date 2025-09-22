# rollups.py
from __future__ import annotations

from pathlib import Path

import polars as pl

ROLLUP_DIR = Path("artifacts/rollups")
ROLLUP_DIR.mkdir(parents=True, exist_ok=True)

GROUP = ["eff_gas_day", "pipeline_name", "state_abb", "category_short"]


def build_daily_rollups(lf: pl.LazyFrame) -> pl.DataFrame:
    receipts = (pl.col("rec_del_sign") == 1).cast(pl.Int8)
    deliveries = (pl.col("rec_del_sign") == -1).cast(pl.Int8)
    agg = (
        lf.group_by(GROUP)
        .agg(
            [
                (pl.when(receipts == 1).then(pl.col("scheduled_quantity")).otherwise(0.0))
                .sum()
                .alias("sum_receipts"),
                (pl.when(deliveries == 1).then(pl.col("scheduled_quantity")).otherwise(0.0))
                .sum()
                .alias("sum_deliveries"),
                pl.col("scheduled_quantity").sum().alias("sum_all"),
            ]
        )
        .with_columns((pl.col("sum_receipts") - pl.col("sum_deliveries")).alias("net"))
    )
    return agg.collect()


def write_daily_rollups(df: pl.DataFrame) -> Path:
    out = ROLLUP_DIR / "daily.parquet"
    df.write_parquet(out)
    return out
