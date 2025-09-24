"""Tests for change point detection and event cards."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from data_agent.core.events import build_event_card, changepoint_detection
from data_agent.core.ops import apply_plan
from data_agent.core.plan_schema import Plan


class TestChangepointDetection:
    """Test change point detection functionality."""

    def test_synthetic_changepoint_single_series(self):
        """Test changepoint detection on synthetic data with known breakpoint."""
        # Create synthetic time series with a clear change point at day 50
        np.random.seed(42)
        n_days = 100
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]

        # Generate data with change point at day 50
        # Before: mean=100, std=10
        # After: mean=200, std=15
        values_before = np.random.normal(100, 10, 50)
        values_after = np.random.normal(200, 15, 50)
        values = np.concatenate([values_before, values_after])

        # Create DataFrame
        df = pl.DataFrame(
            {
                "eff_gas_day": dates,
                "scheduled_quantity": values,
                "pipeline_name": ["TestPipeline"] * n_days,
                "loc_name": ["TestLocation"] * n_days,
            }
        )

        # Run changepoint detection
        result = changepoint_detection(
            df.lazy(),
            groupby_cols=None,
            value_col="scheduled_quantity",
            date_col="eff_gas_day",
            min_size=10,
            penalty=5.0,  # Lower penalty to detect the change point
        )

        assert not result.is_empty(), "Should detect at least one change point"

        # Check that detected changepoint is within ±1 day of the true changepoint (day 50)
        true_changepoint = date(2023, 1, 1) + timedelta(days=50)
        detected_dates = result["changepoint_date"].to_list()

        # Find the closest detected changepoint to the true one
        min_diff = min(abs((d - true_changepoint).days) for d in detected_dates)
        assert (
            min_diff <= 1
        ), f"Changepoint should be detected within ±1 day, but closest was {min_diff} days away"

        # Verify the statistics make sense
        closest_idx = min(
            range(len(detected_dates)),
            key=lambda i: abs((detected_dates[i] - true_changepoint).days),
        )

        row = result.row(closest_idx, named=True)
        assert (
            row["before_mean"] < row["after_mean"]
        ), "After mean should be higher than before mean"
        assert (
            abs(row["before_mean"] - 100) < 30
        ), f"Before mean should be around 100, got {row['before_mean']}"
        assert (
            abs(row["after_mean"] - 200) < 30
        ), f"After mean should be around 200, got {row['after_mean']}"

    def test_synthetic_changepoint_grouped_series(self):
        """Test changepoint detection on grouped data with different breakpoints per group."""
        np.random.seed(42)
        n_days = 80
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]

        data_rows = []

        # Pipeline A: change point at day 40
        values_a_before = np.random.normal(50, 5, 40)
        values_a_after = np.random.normal(150, 10, 40)
        values_a = np.concatenate([values_a_before, values_a_after])

        for d, v in zip(dates, values_a):
            data_rows.append(
                {
                    "eff_gas_day": d,
                    "scheduled_quantity": v,
                    "pipeline_name": "PipelineA",
                    "loc_name": "LocationA",
                }
            )

        # Pipeline B: change point at day 60
        values_b_before = np.random.normal(80, 8, 60)
        values_b_after = np.random.normal(30, 5, 20)
        values_b = np.concatenate([values_b_before, values_b_after])

        for d, v in zip(dates, values_b):
            data_rows.append(
                {
                    "eff_gas_day": d,
                    "scheduled_quantity": v,
                    "pipeline_name": "PipelineB",
                    "loc_name": "LocationB",
                }
            )

        df = pl.DataFrame(data_rows)

        # Run changepoint detection grouped by pipeline
        result = changepoint_detection(
            df.lazy(),
            groupby_cols=["pipeline_name"],
            value_col="scheduled_quantity",
            date_col="eff_gas_day",
            min_size=8,
            penalty=3.0,
        )

        assert not result.is_empty(), "Should detect change points for grouped data"

        # Check Pipeline A changepoint (should be around day 40)
        pipeline_a_results = result.filter(pl.col("pipeline_name") == "PipelineA")
        if not pipeline_a_results.is_empty():
            true_cp_a = date(2023, 1, 1) + timedelta(days=40)
            detected_a = pipeline_a_results["changepoint_date"].to_list()
            min_diff_a = min(abs((d - true_cp_a).days) for d in detected_a)
            assert (
                min_diff_a <= 1
            ), f"Pipeline A changepoint should be within ±1 day, got {min_diff_a}"

        # Check Pipeline B changepoint (should be around day 60)
        pipeline_b_results = result.filter(pl.col("pipeline_name") == "PipelineB")
        if not pipeline_b_results.is_empty():
            true_cp_b = date(2023, 1, 1) + timedelta(days=60)
            detected_b = pipeline_b_results["changepoint_date"].to_list()
            min_diff_b = min(abs((d - true_cp_b).days) for d in detected_b)
            assert (
                min_diff_b <= 1
            ), f"Pipeline B changepoint should be within ±1 day, got {min_diff_b}"

    def test_no_changepoints_stable_series(self):
        """Test that stable time series produces no change points."""
        np.random.seed(42)
        n_days = 100
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]

        # Generate stable series with just noise
        values = np.random.normal(100, 5, n_days)

        df = pl.DataFrame(
            {
                "eff_gas_day": dates,
                "scheduled_quantity": values,
                "pipeline_name": ["TestPipeline"] * n_days,
            }
        )

        result = changepoint_detection(
            df.lazy(),
            groupby_cols=None,
            value_col="scheduled_quantity",
            date_col="eff_gas_day",
            min_size=10,
            penalty=50.0,  # Higher penalty to avoid false positives
        )

        # Should detect few or no change points for stable data
        assert (
            len(result) <= 2
        ), f"Should detect few change points for stable data, got {len(result)}"

    def test_changepoint_via_plan_execution(self):
        """Test changepoint detection through the plan execution system."""
        # Create synthetic data
        np.random.seed(42)
        n_days = 60
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]

        # Change point at day 30
        values_before = np.random.normal(75, 8, 30)
        values_after = np.random.normal(125, 12, 30)
        values = np.concatenate([values_before, values_after])

        df = pl.DataFrame(
            {
                "eff_gas_day": dates,
                "scheduled_quantity": values,
                "pipeline_name": ["TestPipeline"] * n_days,
                "loc_name": ["TestLocation"] * n_days,
            }
        )

        # Create a plan with changepoint operation
        plan = Plan(
            op="changepoint",
            op_args={
                "groupby_cols": None,
                "value_col": "scheduled_quantity",
                "date_col": "eff_gas_day",
                "min_size": 8,
                "penalty": 5.0,
            },
        )

        # Execute the plan
        result_lf = apply_plan(df.lazy(), plan)
        result_df = result_lf.collect()

        assert not result_df.is_empty(), "Plan execution should detect change points"

        # Verify changepoint is detected within tolerance
        true_changepoint = date(2023, 1, 1) + timedelta(days=30)
        detected_dates = result_df["changepoint_date"].to_list()
        min_diff = min(abs((d - true_changepoint).days) for d in detected_dates)
        assert min_diff <= 1, "Changepoint should be detected within ±1 day via plan execution"


class TestEventCard:
    """Test event card generation."""

    def test_build_event_card_with_changepoints(self):
        """Test building event card from changepoint results."""
        # Create mock changepoint results
        changepoints_df = pl.DataFrame(
            {
                "changepoint_date": [date(2023, 2, 15), date(2023, 5, 10)],
                "before_mean": [100.0, 80.0],
                "after_mean": [150.0, 120.0],
                "before_std": [10.0, 8.0],
                "after_std": [15.0, 12.0],
                "change_magnitude": [0.5, 0.5],
                "confidence": [3.5, 2.8],
            }
        )

        # Create mock original data for contributor analysis
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(180)]
        df = pl.DataFrame(
            {
                "eff_gas_day": dates,
                "scheduled_quantity": [100.0] * 180,
                "pipeline_name": ["TestPipeline"] * 180,
                "loc_name": ["TestLocation"] * 180,
            }
        )

        event_card = build_event_card(
            changepoints_df,
            df.lazy(),
            groupby_cols=None,
            date_col="eff_gas_day",
            top_n_contributors=3,
        )

        assert event_card["events_detected"] == 2, "Should detect 2 events"
        assert len(event_card["changepoints"]) == 2, "Should have 2 changepoint entries"

        # Check first changepoint
        first_event = event_card["changepoints"][0]
        assert first_event["date"] == "2023-02-15"
        assert first_event["before_mean"] == 100.0
        assert first_event["after_mean"] == 150.0
        assert "top_contributors" in first_event

    def test_build_event_card_empty_changepoints(self):
        """Test event card generation with no changepoints."""
        empty_df = pl.DataFrame(
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

        # Mock original data
        df = pl.DataFrame(
            {
                "eff_gas_day": [date(2023, 1, 1)],
                "scheduled_quantity": [100.0],
                "pipeline_name": ["TestPipeline"],
                "loc_name": ["TestLocation"],
            }
        )

        event_card = build_event_card(empty_df, df.lazy())

        assert event_card["events_detected"] == 0
        assert len(event_card["changepoints"]) == 0
        assert "No significant change points detected" in event_card["summary"]


if __name__ == "__main__":
    pytest.main([__file__])
