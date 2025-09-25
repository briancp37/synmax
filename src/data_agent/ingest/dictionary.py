"""Data dictionary generation and utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import polars as pl

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def build_data_dictionary(lf: pl.LazyFrame) -> dict[str, Any]:
    """Build a comprehensive data dictionary from a LazyFrame.

    Args:
        lf: Polars LazyFrame to analyze.

    Returns:
        Dictionary with schema, null rates, and row count information.
    """
    # Get schema without collecting data
    schema_info = lf.collect_schema()
    schema = {name: str(dtype) for name, dtype in schema_info.items()}
    column_names = schema_info.names()

    # Compute row count and null rates
    row_count_frame = lf.select([pl.len().alias("n_rows")]).collect()
    n_rows = int(row_count_frame[0, "n_rows"]) if row_count_frame.height > 0 else 0

    if n_rows == 0:
        null_rates = {c: 0.0 for c in column_names}
    else:
        # Compute null rates
        nulls_frame = (
            lf.select([pl.col(c).is_null().sum().alias(c) for c in column_names])
            .with_columns(pl.all().cast(pl.Float64) / max(n_rows, 1))
            .collect()
        )
        null_rates = {c: float(nulls_frame[0, c]) for c in column_names}

    return {"schema": schema, "null_rates": null_rates, "n_rows": n_rows}


def write_dictionary(dic: dict[str, Any], path: Path = ARTIFACTS / "data_dictionary.json") -> None:
    """Write data dictionary to JSON file.

    Args:
        dic: Dictionary to write.
        path: Output path for JSON file.
    """
    path.write_bytes(orjson.dumps(dic, option=orjson.OPT_INDENT_2))


def load_data_dictionary(path: Path = ARTIFACTS / "data_dictionary.json") -> dict[str, Any]:
    """Load data dictionary from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Dictionary with schema, null rates, and row count information.

    Raises:
        FileNotFoundError: If dictionary file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Data dictionary not found at {path}. Run 'agent load' first to generate it."
        )
    
    return orjson.loads(path.read_bytes())
