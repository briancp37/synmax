"""Tests for rollups functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from data_agent.ingest.rollups import build_daily_rollups, write_daily_rollups


def test_build_daily_rollups():
    """Test that daily rollups are built correctly."""
    # Create test data
    test_data = pl.DataFrame(
        {
            "eff_gas_day": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "pipeline_name": ["Pipeline A", "Pipeline A", "Pipeline B", "Pipeline A", "Pipeline B"],
            "state_abb": ["TX", "TX", "LA", "TX", "LA"],
            "category_short": ["Interconnect", "LDC", "Industrial", "Interconnect", "Industrial"],
            "rec_del_sign": [1, -1, 1, -1, 1],  # receipts=1, deliveries=-1
            "scheduled_quantity": [100.0, 50.0, 75.0, 25.0, 80.0],
        }
    ).with_columns(pl.col("eff_gas_day").str.strptime(pl.Date))

    lf = test_data.lazy()

    # Build rollups
    result = build_daily_rollups(lf)

    # Verify structure
    expected_columns = {
        "eff_gas_day",
        "pipeline_name",
        "state_abb",
        "category_short",
        "sum_receipts",
        "sum_deliveries",
        "sum_all",
        "net",
    }
    assert set(result.columns) == expected_columns

    # Convert to list of dicts for easier testing
    result_rows = result.to_dicts()

    # Should have 5 rows (one per unique combination of group columns)
    assert len(result_rows) == 5

    # Check specific calculations for one group
    pipeline_a_tx_interconnect = [
        row
        for row in result_rows
        if (
            row["pipeline_name"] == "Pipeline A"
            and row["state_abb"] == "TX"
            and row["category_short"] == "Interconnect"
        )
    ]

    # Should have two rows (one for each day)
    assert len(pipeline_a_tx_interconnect) == 2

    # Find the 2023-01-01 row
    jan_1_row = next(
        row
        for row in pipeline_a_tx_interconnect
        if row["eff_gas_day"].strftime("%Y-%m-%d") == "2023-01-01"
    )

    # On 2023-01-01, Pipeline A TX Interconnect had:
    # - 1 receipt of 100.0 (rec_del_sign=1)
    # - 0 deliveries
    assert jan_1_row["sum_receipts"] == 100.0
    assert jan_1_row["sum_deliveries"] == 0.0
    assert jan_1_row["sum_all"] == 100.0
    assert jan_1_row["net"] == 100.0

    # Find the 2023-01-02 row
    jan_2_row = next(
        row
        for row in pipeline_a_tx_interconnect
        if row["eff_gas_day"].strftime("%Y-%m-%d") == "2023-01-02"
    )

    # On 2023-01-02, Pipeline A TX Interconnect had:
    # - 0 receipts
    # - 1 delivery of 25.0 (rec_del_sign=-1)
    assert jan_2_row["sum_receipts"] == 0.0
    assert jan_2_row["sum_deliveries"] == 25.0
    assert jan_2_row["sum_all"] == 25.0
    assert jan_2_row["net"] == -25.0


def test_write_daily_rollups():
    """Test that rollups are written to correct path."""
    # Create test rollups DataFrame
    test_rollups = pl.DataFrame(
        {
            "eff_gas_day": ["2023-01-01"],
            "pipeline_name": ["Test Pipeline"],
            "state_abb": ["TX"],
            "category_short": ["LDC"],
            "sum_receipts": [100.0],
            "sum_deliveries": [50.0],
            "sum_all": [150.0],
            "net": [50.0],
        }
    ).with_columns(pl.col("eff_gas_day").str.strptime(pl.Date))

    # Write rollups
    with tempfile.TemporaryDirectory() as tmpdir:
        # Monkey patch ROLLUP_DIR for this test
        import data_agent.ingest.rollups as rollups_module

        original_rollup_dir = rollups_module.ROLLUP_DIR
        rollups_module.ROLLUP_DIR = Path(tmpdir) / "rollups"
        rollups_module.ROLLUP_DIR.mkdir(parents=True, exist_ok=True)

        try:
            result_path = write_daily_rollups(test_rollups)

            # Verify path
            expected_path = rollups_module.ROLLUP_DIR / "daily.parquet"
            assert result_path == expected_path
            assert result_path.exists()

            # Verify content can be read back
            loaded = pl.read_parquet(result_path)
            assert loaded.equals(test_rollups)

        finally:
            # Restore original ROLLUP_DIR
            rollups_module.ROLLUP_DIR = original_rollup_dir


def test_rollups_with_missing_columns():
    """Test rollups behavior with missing required columns."""
    # Test data missing 'category_short'
    test_data = pl.DataFrame(
        {
            "eff_gas_day": ["2023-01-01"],
            "pipeline_name": ["Pipeline A"],
            "state_abb": ["TX"],
            "rec_del_sign": [1],
            "scheduled_quantity": [100.0],
        }
    ).with_columns(pl.col("eff_gas_day").str.strptime(pl.Date))

    lf = test_data.lazy()

    # Should raise an error due to missing column
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        build_daily_rollups(lf)


def test_golden_rollups_deterministic():
    """Test that rollups for golden dataset match expected output."""
    # Load golden dataset
    golden_path = Path("examples/golden.parquet")
    if not golden_path.exists():
        pytest.skip("Golden dataset not found")

    lf = pl.scan_parquet(str(golden_path))

    # Build rollups
    result = build_daily_rollups(lf)

    # Sort for deterministic comparison
    result = result.sort(["eff_gas_day", "pipeline_name", "state_abb", "category_short"])

    # Convert to dict for JSON comparison
    result_data = result.to_dicts()

    # Convert dates to strings for JSON serialization
    for row in result_data:
        row["eff_gas_day"] = row["eff_gas_day"].strftime("%Y-%m-%d")

    # Load expected output
    expected_path = Path("examples/expected/daily.json")
    if not expected_path.exists():
        pytest.skip("Expected rollups output not found")

    with open(expected_path) as f:
        expected_data = json.load(f)

    # Compare
    assert len(result_data) == len(
        expected_data
    ), f"Expected {len(expected_data)} rows, got {len(result_data)}"

    # Compare each row
    for i, (result_row, expected_row) in enumerate(zip(result_data, expected_data)):
        assert result_row == expected_row, f"Mismatch at row {i}: {result_row} != {expected_row}"
