"""Tests for metrics functions."""

from __future__ import annotations

import polars as pl
import pytest

from data_agent.core.metrics import imbalance_pct, ramp_risk, reversal_freq


class TestRampRisk:
    """Test ramp_risk function."""

    def test_ramp_risk_basic(self):
        """Test basic ramp risk calculation."""
        # Create toy data with known deltas
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A", "A", "A"],
                "loc_name": ["L1", "L1", "L1", "L1"],
                "eff_gas_day": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
                "scheduled_quantity": [100.0, 110.0, 90.0, 100.0],  # deltas: 10, 20, 10
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = ramp_risk(df.lazy()).collect()

        assert len(result) == 1
        assert result["pipeline_name"][0] == "A"
        assert result["loc_name"][0] == "L1"
        assert result["n_days"][0] == 4

        # p95 of [10, 20, 10] = 20, median of [100, 110, 90, 100] = 100
        # ramp_risk = 20 / 100 = 0.2
        assert result["p95_abs_delta"][0] == 20.0
        assert result["median_flow"][0] == 100.0
        assert result["ramp_risk"][0] == pytest.approx(0.2)

    def test_ramp_risk_multiple_entities(self):
        """Test ramp risk with multiple pipeline/location combinations."""
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A", "B", "B"],
                "loc_name": ["L1", "L1", "L2", "L2"],
                "eff_gas_day": ["2022-01-01", "2022-01-02", "2022-01-01", "2022-01-02"],
                "scheduled_quantity": [100.0, 120.0, 50.0, 55.0],
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = ramp_risk(df.lazy()).collect().sort(["pipeline_name", "loc_name"])

        assert len(result) == 2
        assert result["pipeline_name"].to_list() == ["A", "B"]
        assert result["loc_name"].to_list() == ["L1", "L2"]

    def test_ramp_risk_zero_median(self):
        """Test ramp risk handles zero median flow."""
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A", "A"],
                "loc_name": ["L1", "L1", "L1"],
                "eff_gas_day": ["2022-01-01", "2022-01-02", "2022-01-03"],
                "scheduled_quantity": [0.0, 0.0, 10.0],  # median = 0.0
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = ramp_risk(df.lazy()).collect()

        # Should use max(median_flow, 1.0) = 1.0 as denominator
        # p95 of deltas [0, 10] = 10, so ramp_risk = 10/1 = 10
        assert result["ramp_risk"][0] == 10.0


class TestReversalFreq:
    """Test reversal_freq function."""

    def test_reversal_freq_basic(self):
        """Test basic reversal frequency calculation."""
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A", "A", "A"],
                "connecting_entity": ["E1", "E1", "E1", "E1"],
                "eff_gas_day": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
                "rec_del_sign": [1, 1, -1, -1],  # Receipt, Receipt, Delivery, Delivery
                "scheduled_quantity": [100.0, 50.0, 75.0, 25.0],
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = reversal_freq(df.lazy()).collect()

        assert len(result) == 1
        assert result["pipeline_name"][0] == "A"
        assert result["connecting_entity"][0] == "E1"
        assert result["n_days"][0] == 4

        # Net flows: [100, 50, -75, -25], signs: [+, +, -, -]
        # Reversals: day 3 (+ to -)
        assert result["n_reversals"][0] == 1
        assert result["reversal_freq"][0] == pytest.approx(0.25)

    def test_reversal_freq_no_reversals(self):
        """Test reversal frequency with no sign changes."""
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A", "A"],
                "connecting_entity": ["E1", "E1", "E1"],
                "eff_gas_day": ["2022-01-01", "2022-01-02", "2022-01-03"],
                "rec_del_sign": [1, 1, 1],
                "scheduled_quantity": [100.0, 50.0, 75.0],
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = reversal_freq(df.lazy()).collect()

        assert result["n_reversals"][0] == 0
        assert result["reversal_freq"][0] == 0.0

    def test_reversal_freq_multiple_entities(self):
        """Test reversal frequency with multiple connections."""
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A", "B", "B"],
                "connecting_entity": ["E1", "E1", "E2", "E2"],
                "eff_gas_day": ["2022-01-01", "2022-01-02", "2022-01-01", "2022-01-02"],
                "rec_del_sign": [1, -1, 1, 1],
                "scheduled_quantity": [100.0, 50.0, 75.0, 25.0],
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = reversal_freq(df.lazy()).collect().sort(["pipeline_name", "connecting_entity"])

        assert len(result) == 2
        assert result["n_reversals"].to_list() == [1, 0]  # A-E1 has reversal, B-E2 doesn't


class TestImbalancePct:
    """Test imbalance_pct function."""

    def test_imbalance_pct_basic(self):
        """Test basic imbalance percentage calculation."""
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A", "A"],
                "eff_gas_day": ["2022-01-01", "2022-01-01", "2022-01-01"],
                "rec_del_sign": [1, 1, -1],  # Two receipts, one delivery
                "scheduled_quantity": [100.0, 50.0, 75.0],
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = imbalance_pct(df.lazy()).collect()

        assert len(result) == 1
        assert result["pipeline_name"][0] == "A"
        assert result["total_receipts"][0] == 150.0
        assert result["total_deliveries"][0] == 75.0
        assert result["net_flow"][0] == 75.0

        # Imbalance = |150 - 75| / 150 * 100 = 50%
        assert result["imbalance_pct"][0] == pytest.approx(50.0)

    def test_imbalance_pct_multiple_days(self):
        """Test imbalance percentage across multiple days."""
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A", "A", "A"],
                "eff_gas_day": ["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-02"],
                "rec_del_sign": [1, -1, 1, -1],
                "scheduled_quantity": [100.0, 100.0, 50.0, 60.0],
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = imbalance_pct(df.lazy()).collect().sort("eff_gas_day")

        assert len(result) == 2

        # Day 1: receipts=100, deliveries=100, imbalance=0%
        assert result["imbalance_pct"][0] == 0.0

        # Day 2: receipts=50, deliveries=60, imbalance=|50-60|/50*100=20%
        assert result["imbalance_pct"][1] == pytest.approx(20.0)

    def test_imbalance_pct_zero_receipts(self):
        """Test imbalance percentage handles zero receipts."""
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A"],
                "eff_gas_day": ["2022-01-01", "2022-01-01"],
                "rec_del_sign": [-1, -1],
                "scheduled_quantity": [50.0, 25.0],
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = imbalance_pct(df.lazy()).collect()

        # Should use max(receipts, 1.0) = 1.0 as denominator
        assert result["total_receipts"][0] == 0.0
        assert result["imbalance_pct"][0] == pytest.approx(7500.0)  # 75/1*100

    def test_imbalance_pct_multiple_pipelines(self):
        """Test imbalance percentage with multiple pipelines."""
        df = pl.DataFrame(
            {
                "pipeline_name": ["A", "A", "B", "B"],
                "eff_gas_day": ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"],
                "rec_del_sign": [1, -1, 1, -1],
                "scheduled_quantity": [100.0, 50.0, 200.0, 200.0],
            }
        ).with_columns(pl.col("eff_gas_day").str.to_date())

        result = imbalance_pct(df.lazy()).collect().sort("pipeline_name")

        assert len(result) == 2
        assert result["pipeline_name"].to_list() == ["A", "B"]
        assert result["imbalance_pct"][0] == pytest.approx(50.0)  # A: |100-50|/100*100
        assert result["imbalance_pct"][1] == pytest.approx(0.0)  # B: |200-200|/200*100
