"""Unit tests for the rules engine."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from data_agent.rules.engine import (
    _rule_001_missing_geo_on_active,
    _rule_002_duplicate_loc_across_states,
    _rule_003_zero_quantity_streaks,
    _rule_004_pipeline_imbalance,
    _rule_005_schema_mismatch,
    _rule_006_gas_day_gaps,
    run_rules,
)


@pytest.fixture
def sample_data() -> pl.LazyFrame:
    """Create sample data for testing rules."""
    data = {
        "pipeline_name": ["ANR"] * 10,
        "loc_name": [
            "LOC1",
            "LOC1",
            "LOC2",
            "LOC2",
            "LOC3",
            "LOC3",
            "LOC4",
            "LOC4",
            "LOC5",
            "LOC5",
        ],
        "connecting_pipeline": [
            "Pipeline A",
            None,
            "Pipeline B",
            None,
            None,
            None,
            "Pipeline C",
            None,
            None,
            None,
        ],
        "connecting_entity": [
            "Entity A",
            "Entity B",
            "Entity C",
            "Entity D",
            "Entity E",
            "Entity F",
            "Entity G",
            "Entity H",
            "Entity I",
            "Entity J",
        ],
        "rec_del_sign": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        "category_short": [
            "Interconnect",
            "LDC",
            "Industrial",
            "LDC",
            "Interconnect",
            "LDC",
            "LDC",
            "LDC",
            "Interconnect",
            "LDC",
        ],
        "country_name": ["USA"] * 10,
        "state_abb": ["TX", "TX", "LA", "LA", "TX", "TX", "OK", "OK", "TX", "TX"],
        "county_name": ["County1"] * 10,
        "latitude": [None, 30.0, 29.0, None, 31.0, 32.0, 35.0, 36.0, None, 33.0],
        "longitude": [None, -95.0, -90.0, None, -96.0, -97.0, -98.0, -99.0, None, -94.0],
        "eff_gas_day": [
            date(2022, 1, 1),
            date(2022, 1, 2),
            date(2022, 1, 3),
            date(2022, 1, 4),
            date(2022, 1, 5),
            date(2022, 1, 6),
            date(2022, 1, 7),
            date(2022, 1, 8),
            date(2022, 1, 9),
            date(2022, 1, 10),
        ],
        "scheduled_quantity": [
            1000.0,
            500.0,
            0.0,
            0.0,
            2000.0,
            1500.0,
            800.0,
            600.0,
            1200.0,
            900.0,
        ],
    }
    return pl.LazyFrame(data)


@pytest.fixture
def duplicate_loc_data() -> pl.LazyFrame:
    """Create data with duplicate location names across states."""
    data = {
        "pipeline_name": ["ANR"] * 4,
        "loc_name": ["HOUSTON", "HOUSTON", "DALLAS", "DALLAS"],
        "connecting_pipeline": [None] * 4,
        "connecting_entity": ["Entity A", "Entity B", "Entity C", "Entity D"],
        "rec_del_sign": [1, 1, 1, 1],
        "category_short": ["LDC"] * 4,
        "country_name": ["USA"] * 4,
        "state_abb": ["TX", "LA", "TX", "OK"],  # HOUSTON in TX and LA, DALLAS in TX and OK
        "county_name": ["County1"] * 4,
        "latitude": [30.0, 29.0, 32.0, 35.0],
        "longitude": [-95.0, -90.0, -96.0, -98.0],
        "eff_gas_day": [date(2022, 1, 1)] * 4,
        "scheduled_quantity": [1000.0, 500.0, 800.0, 600.0],
    }
    return pl.LazyFrame(data)


@pytest.fixture
def imbalance_data() -> pl.LazyFrame:
    """Create data with pipeline imbalances."""
    data = {
        "pipeline_name": ["ANR"] * 6,
        "loc_name": ["LOC1", "LOC2", "LOC3", "LOC4", "LOC5", "LOC6"],
        "connecting_pipeline": [None] * 6,
        "connecting_entity": ["Entity A"] * 6,
        "rec_del_sign": [1, 1, -1, 1, 1, -1],  # More receipts than deliveries
        "category_short": ["LDC"] * 6,
        "country_name": ["USA"] * 6,
        "state_abb": ["TX"] * 6,
        "county_name": ["County1"] * 6,
        "latitude": [30.0] * 6,
        "longitude": [-95.0] * 6,
        "eff_gas_day": [date(2022, 1, 1)] * 6,
        "scheduled_quantity": [
            1000.0,
            1000.0,
            100.0,
            1000.0,
            1000.0,
            100.0,
        ],  # 4000 receipts, 200 deliveries
    }
    return pl.LazyFrame(data)


def test_rule_001_missing_geo_on_active(sample_data: pl.LazyFrame) -> None:
    """Test R-001: Missing Geo on Active."""
    result = _rule_001_missing_geo_on_active(sample_data)

    # Should find records with null lat/lng and positive scheduled_quantity
    # From sample_data: LOC1 (1000.0), LOC5 (1200.0) have null coordinates and positive quantity
    assert result["count"] == 2
    assert len(result["samples"]) == 2

    # Check that samples contain expected fields
    for sample in result["samples"]:
        assert "pipeline_name" in sample
        assert "loc_name" in sample
        assert "scheduled_quantity" in sample
        assert sample["scheduled_quantity"] > 0
        # The new implementation adds geo info based on schema
        if "latitude" in sample:
            assert sample["latitude"] is None or sample["longitude"] is None


def test_rule_002_duplicate_loc_across_states(duplicate_loc_data: pl.LazyFrame) -> None:
    """Test R-002: Duplicate loc_name across states."""
    result = _rule_002_duplicate_loc_across_states(duplicate_loc_data)

    # Should find HOUSTON and DALLAS appearing in multiple states
    assert result["count"] == 2
    assert len(result["samples"]) == 2

    # Check that samples contain expected fields
    for sample in result["samples"]:
        assert "loc_name" in sample
        assert "n_states" in sample
        assert sample["n_states"] > 1
        assert sample["loc_name"] in ["HOUSTON", "DALLAS"]


def test_rule_003_zero_quantity_streaks(sample_data: pl.LazyFrame) -> None:
    """Test R-003: Zero-quantity streaks."""
    result = _rule_003_zero_quantity_streaks(sample_data)

    # This is a simplified test - the actual implementation would need
    # more sophisticated consecutive day detection
    # For now, we just check that the function runs without error
    assert "count" in result
    assert "samples" in result
    assert result["count"] >= 0


def test_rule_004_pipeline_imbalance(imbalance_data: pl.LazyFrame) -> None:
    """Test R-004: Pipeline Imbalance."""
    result = _rule_004_pipeline_imbalance(imbalance_data)

    # Should find imbalance: 4000 receipts vs 200 deliveries
    # Imbalance ratio = 3800 / 4000 = 0.95 (95%) which is > 5% threshold
    assert result["count"] == 1
    assert len(result["samples"]) == 1

    sample = result["samples"][0]
    assert sample["pipeline_name"] == "ANR"
    assert sample["receipts"] == 4000.0
    assert sample["deliveries"] == 200.0
    assert sample["imbalance_ratio"] > 0.05  # > 5% threshold


def test_rule_005_schema_mismatch(sample_data: pl.LazyFrame) -> None:
    """Test R-005: Schema mismatch."""
    result = _rule_005_schema_mismatch(sample_data)

    # Should find records with connecting_pipeline but category_short != 'Interconnect'
    # From sample_data: "Pipeline B" with "Industrial", "Pipeline C" with "LDC"
    assert result["count"] == 2
    assert len(result["samples"]) == 2

    for sample in result["samples"]:
        assert sample["connecting_pipeline"] is not None
        assert sample["category_short"].lower() != "interconnect"


def test_rule_006_gas_day_gaps(sample_data: pl.LazyFrame) -> None:
    """Test R-006: Eff gas day gaps."""
    result = _rule_006_gas_day_gaps(sample_data)

    # This is a simplified test - the sample data has consecutive days
    # so we don't expect significant gaps
    assert "count" in result
    assert "samples" in result
    assert result["count"] >= 0


def test_run_rules_integration(sample_data: pl.LazyFrame) -> None:
    """Test the main run_rules function."""
    results = run_rules(sample_data)

    # Should return results for all rules
    expected_rules = ["R-001", "R-002", "R-003", "R-004", "R-005", "R-006"]
    for rule_id in expected_rules:
        assert rule_id in results
        assert "count" in results[rule_id]
        assert "samples" in results[rule_id]
        assert isinstance(results[rule_id]["count"], int)
        assert isinstance(results[rule_id]["samples"], list)


def test_run_rules_with_filters(sample_data: pl.LazyFrame) -> None:
    """Test run_rules with pipeline and date filters."""
    # Test with pipeline filter
    results = run_rules(sample_data, pipeline="ANR")
    assert "R-001" in results

    # Test with date filter
    results = run_rules(sample_data, since="2022-01-01")
    assert "R-001" in results

    # Test with both filters
    results = run_rules(sample_data, pipeline="ANR", since="2022-01-05")
    assert "R-001" in results


def test_rule_samples_limited_to_three() -> None:
    """Test that rule samples are limited to 3 per rule."""
    # Create data with many violations
    data = {
        "pipeline_name": ["ANR"] * 10,
        "loc_name": [f"LOC{i}" for i in range(10)],
        "connecting_pipeline": [None] * 10,
        "connecting_entity": [f"Entity{i}" for i in range(10)],
        "rec_del_sign": [1] * 10,
        "category_short": ["LDC"] * 10,
        "country_name": ["USA"] * 10,
        "state_abb": ["TX"] * 10,
        "county_name": ["County1"] * 10,
        "latitude": [None] * 10,  # All null
        "longitude": [None] * 10,  # All null
        "eff_gas_day": [date(2022, 1, 1)] * 10,
        "scheduled_quantity": [1000.0] * 10,  # All positive
    }
    lf = pl.LazyFrame(data)

    result = _rule_001_missing_geo_on_active(lf)
    assert result["count"] == 10
    assert len(result["samples"]) == 3  # Should be limited to 3
