"""Integration tests for metrics functions using golden.parquet."""

from __future__ import annotations

import polars as pl
import pytest

from data_agent.core.metrics import imbalance_pct, ramp_risk, reversal_freq
from data_agent.core.ops import apply_plan
from data_agent.core.plan_schema import Filter, Plan


class TestMetricsIntegration:
    """Integration tests using golden.parquet dataset."""

    @pytest.fixture
    def golden_df(self):
        """Load the golden dataset."""
        return pl.scan_parquet("examples/golden.parquet")

    def test_ramp_risk_integration(self, golden_df):
        """Test ramp_risk on golden dataset."""
        # Apply some filters to get a manageable subset
        plan = Plan(
            filters=[
                Filter(column="pipeline_name", op="=", value="ANR Pipeline Company"),
                Filter(column="eff_gas_day", op="between", value=["2022-01-01", "2022-03-31"]),
            ]
        )

        filtered_lf = apply_plan(golden_df, plan)
        result = ramp_risk(filtered_lf).collect()

        # Verify structure
        assert "pipeline_name" in result.columns
        assert "loc_name" in result.columns
        assert "ramp_risk" in result.columns
        assert "n_days" in result.columns
        assert "p95_abs_delta" in result.columns
        assert "median_flow" in result.columns

        # Verify data types
        assert result["ramp_risk"].dtype == pl.Float64
        assert result["n_days"].dtype == pl.UInt32

        # Verify non-negative values
        assert all(result["ramp_risk"] >= 0)
        assert all(result["n_days"] > 0)

        # Should have results for multiple locations
        assert len(result) > 0
        print(f"Ramp risk results: {len(result)} locations")
        print(result.head())

    def test_reversal_freq_integration(self, golden_df):
        """Test reversal_freq on golden dataset."""
        # Apply filters to get subset with potential reversals
        plan = Plan(
            filters=[
                Filter(column="pipeline_name", op="=", value="ANR Pipeline Company"),
                Filter(column="eff_gas_day", op="between", value=["2022-01-01", "2022-03-31"]),
            ]
        )

        filtered_lf = apply_plan(golden_df, plan)
        result = reversal_freq(filtered_lf).collect()

        # Verify structure
        assert "pipeline_name" in result.columns
        assert "connecting_entity" in result.columns
        assert "reversal_freq" in result.columns
        assert "n_days" in result.columns
        assert "n_reversals" in result.columns
        assert "net_flow_std" in result.columns

        # Verify data types
        assert result["reversal_freq"].dtype == pl.Float64
        assert result["n_days"].dtype == pl.UInt32
        assert result["n_reversals"].dtype == pl.UInt32

        # Verify ranges
        assert all(result["reversal_freq"] >= 0)
        assert all(result["reversal_freq"] <= 1)
        assert all(result["n_days"] > 0)
        assert all(result["n_reversals"] >= 0)

        # Should have results
        assert len(result) > 0
        print(f"Reversal frequency results: {len(result)} connections")
        print(result.head())

    def test_imbalance_pct_integration(self, golden_df):
        """Test imbalance_pct on golden dataset."""
        # Apply filters to get subset
        plan = Plan(
            filters=[
                Filter(column="pipeline_name", op="=", value="ANR Pipeline Company"),
                Filter(column="eff_gas_day", op="between", value=["2022-01-01", "2022-01-31"]),
            ]
        )

        filtered_lf = apply_plan(golden_df, plan)
        result = imbalance_pct(filtered_lf).collect()

        # Verify structure
        assert "pipeline_name" in result.columns
        assert "eff_gas_day" in result.columns
        assert "imbalance_pct" in result.columns
        assert "total_receipts" in result.columns
        assert "total_deliveries" in result.columns
        assert "net_flow" in result.columns

        # Verify data types
        assert result["imbalance_pct"].dtype == pl.Float64
        assert result["total_receipts"].dtype == pl.Float64
        assert result["total_deliveries"].dtype == pl.Float64

        # Verify non-negative values for quantities
        assert all(result["total_receipts"] >= 0)
        assert all(result["total_deliveries"] >= 0)
        assert all(result["imbalance_pct"] >= 0)

        # Should have daily results
        assert len(result) > 0
        print(f"Imbalance results: {len(result)} days")
        print(result.head())

    def test_metric_compute_via_ops(self, golden_df):
        """Test metrics via the ops.apply_plan interface."""
        # Test ramp_risk via metric_compute operation
        plan = Plan(
            filters=[
                Filter(column="pipeline_name", op="=", value="ANR Pipeline Company"),
                Filter(column="eff_gas_day", op="between", value=["2022-01-01", "2022-03-31"]),
            ],
            op="metric_compute",
            op_args={"name": "ramp_risk"},
        )

        result = apply_plan(golden_df, plan).collect()

        # Should have ramp_risk results
        assert "ramp_risk" in result.columns
        assert len(result) > 0

        # Test reversal_freq via metric_compute operation
        plan = Plan(
            filters=[
                Filter(column="pipeline_name", op="=", value="ANR Pipeline Company"),
                Filter(column="eff_gas_day", op="between", value=["2022-01-01", "2022-03-31"]),
            ],
            op="metric_compute",
            op_args={"name": "reversal_freq"},
        )

        result = apply_plan(golden_df, plan).collect()

        # Should have reversal_freq results
        assert "reversal_freq" in result.columns
        assert len(result) > 0

        # Test imbalance_pct via metric_compute operation
        plan = Plan(
            filters=[
                Filter(column="pipeline_name", op="=", value="ANR Pipeline Company"),
                Filter(column="eff_gas_day", op="between", value=["2022-01-01", "2022-01-31"]),
            ],
            op="metric_compute",
            op_args={"name": "imbalance_pct"},
        )

        result = apply_plan(golden_df, plan).collect()

        # Should have imbalance_pct results
        assert "imbalance_pct" in result.columns
        assert len(result) > 0

    def test_unknown_metric_error(self, golden_df):
        """Test that unknown metric names raise errors."""
        plan = Plan(op="metric_compute", op_args={"name": "unknown_metric"})

        with pytest.raises(ValueError, match="Unknown metric: unknown_metric"):
            apply_plan(golden_df, plan).collect()

    def test_metrics_respect_filters(self, golden_df):
        """Test that metrics respect applied filters."""
        # Test with no filters
        plan_all = Plan(op="metric_compute", op_args={"name": "ramp_risk"})
        result_all = apply_plan(golden_df, plan_all).collect()

        # Test with filters
        plan_filtered = Plan(
            filters=[Filter(column="pipeline_name", op="=", value="ANR Pipeline Company")],
            op="metric_compute",
            op_args={"name": "ramp_risk"},
        )
        result_filtered = apply_plan(golden_df, plan_filtered).collect()

        # Filtered result should be smaller or equal
        assert len(result_filtered) <= len(result_all)

        # All results should be for ANR Pipeline Company only
        unique_pipelines = result_filtered["pipeline_name"].unique().to_list()
        assert unique_pipelines == ["ANR Pipeline Company"]

    def test_metrics_output_tidy_tables(self, golden_df):
        """Test that metrics output well-structured tidy tables."""
        plan = Plan(
            filters=[
                Filter(
                    column="pipeline_name",
                    op="in",
                    value=["ANR Pipeline Company", "Columbia Gulf Transmission Company"],
                )
            ],
            op="metric_compute",
            op_args={"name": "imbalance_pct"},
        )

        result = apply_plan(golden_df, plan).collect()

        # Should be tidy: one row per pipeline-day combination
        assert len(result) > 0

        # Check for duplicates (should be none in tidy format)
        grouped = result.group_by(["pipeline_name", "eff_gas_day"]).len()
        assert all(grouped["len"] == 1)  # Each combination should appear exactly once

        # Columns should be well-named and typed
        expected_cols = [
            "pipeline_name",
            "eff_gas_day",
            "imbalance_pct",
            "total_receipts",
            "total_deliveries",
            "net_flow",
        ]
        for col in expected_cols:
            assert col in result.columns

        # Should be sorted by pipeline and date
        sorted_result = result.sort(["pipeline_name", "eff_gas_day"])
        assert result.equals(sorted_result)
