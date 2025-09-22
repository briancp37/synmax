"""Dataset loader with auto-download and dtype normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

DEFAULT_DATA_PATH = Path("data/data.parquet")


def load_dataset(path: str | None, auto: bool) -> pl.LazyFrame:
    """Return a Polars LazyFrame for the dataset. Does not collect.

    Args:
        path: Path to parquet file. If None, uses DEFAULT_DATA_PATH.
        auto: If True and path not found, attempt to download dataset.

    Returns:
        LazyFrame with normalized dtypes.

    Raises:
        FileNotFoundError: If path doesn't exist and auto=False.
    """
    p = Path(path) if path else DEFAULT_DATA_PATH

    if not p.exists():
        if not auto:
            raise FileNotFoundError(f"{p} not found; pass --auto to download")
        # TODO: download via requests/gdown into p
        # For now, we'll assume the user provides the data
        raise FileNotFoundError(f"{p} not found and auto-download not implemented yet")

    lf = pl.scan_parquet(str(p))

    # dtype normalization
    casts: dict[str, Any] = {
        "rec_del_sign": pl.Int8,
        "scheduled_quantity": pl.Float64,
    }

    # Get column names efficiently
    column_names = lf.collect_schema().names()

    # Handle eff_gas_day -> Date; handle string or date
    if "eff_gas_day" in column_names:
        # Check if it's already a date type by trying to use it
        try:
            # If it's already a date, this will work fine
            lf = lf.with_columns(
                pl.col("eff_gas_day").cast(pl.Date, strict=False).alias("eff_gas_day")
            )
        except Exception:
            # If it's a string, parse it
            lf = lf.with_columns(
                pl.col("eff_gas_day").str.strptime(pl.Date, strict=False).alias("eff_gas_day")
            )

    # Apply casts for other columns
    lf = lf.with_columns(
        [pl.col(k).cast(v, strict=False) for k, v in casts.items() if k in column_names]
    )

    return lf
