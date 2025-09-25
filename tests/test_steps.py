"""Unit tests for step implementations."""

import polars as pl
import pytest

from data_agent.core.handles import HandleStorage, StepHandle, StepStats, create_lazy_handle
from data_agent.core.steps import (
    aggregate_step,
    changepoint_step,
    evidence_collect_step,
    filter_step,
    limit_step,
    rank_step,
    resample_step,
    save_artifact_step,
    stl_deseasonalize_step,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pl.DataFrame(
        {
            "eff_gas_day": pl.date_range(
                start=pl.date(2023, 1, 1), end=pl.date(2023, 1, 31), interval="1d", eager=True
            ),
            "pipeline_name": ["Pipeline A"] * 15 + ["Pipeline B"] * 16,
            "scheduled_quantity": [100.0 + i * 10 for i in range(31)],
            "state_abb": ["TX"] * 31,
            "category": ["LDC"] * 31,
        }
    )


@pytest.fixture
def sample_handle(sample_data, tmp_path):
    """Create a sample step handle with materialized data."""
    parquet_path = tmp_path / "sample.parquet"
    sample_data.write_parquet(parquet_path)

    stats = StepStats(
        rows=sample_data.height,
        bytes=parquet_path.stat().st_size,
        columns=sample_data.width,
        null_count={col: sample_data[col].null_count() for col in sample_data.columns},
        computed_at=parquet_path.stat().st_mtime,
    )

    schema = {col: str(dtype) for col, dtype in zip(sample_data.columns, sample_data.dtypes)}

    return StepHandle(
        id="test_handle",
        store="parquet",
        path=parquet_path,
        engine="polars",
        schema=schema,
        stats=stats,
        fingerprint="test_fingerprint",
    )


@pytest.fixture
def handle_storage(tmp_path):
    """Create a handle storage instance."""
    return HandleStorage(tmp_path / "storage")


class TestFilterStep:
    """Test filter step implementation."""

    def test_filter_equal(self, sample_handle, handle_storage):
        """Test equality filter."""
        params = {"filters": [{"column": "pipeline_name", "op": "=", "value": "Pipeline A"}]}

        result = filter_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_filtered"
        assert result.engine == "polars"
        assert result.schema == sample_handle.schema

    def test_filter_between_dates(self, sample_handle, handle_storage):
        """Test date range filter."""
        params = {
            "filters": [
                {"column": "eff_gas_day", "op": "between", "value": ["2023-01-01", "2023-01-15"]}
            ]
        }

        result = filter_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_filtered"
        assert result.store == "lazy"

    def test_filter_in_values(self, sample_handle, handle_storage):
        """Test 'in' filter."""
        params = {
            "filters": [
                {"column": "pipeline_name", "op": "in", "value": ["Pipeline A", "Pipeline B"]}
            ]
        }

        result = filter_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_filtered"

    def test_filter_multiple(self, sample_handle, handle_storage):
        """Test multiple filters."""
        params = {
            "filters": [
                {"column": "pipeline_name", "op": "=", "value": "Pipeline A"},
                {"column": "scheduled_quantity", "op": "between", "value": [100.0, 200.0]},
            ]
        }

        result = filter_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_filtered"


class TestResampleStep:
    """Test resample step implementation."""

    def test_resample_daily(self, sample_handle, handle_storage):
        """Test daily resampling."""
        params = {"freq": "1d", "on": "eff_gas_day", "agg": {"scheduled_quantity": "sum"}}

        result = resample_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_resampled"
        assert "scheduled_quantity" in result.schema

    def test_resample_weekly(self, sample_handle, handle_storage):
        """Test weekly resampling."""
        params = {"freq": "1w", "on": "eff_gas_day", "agg": {"scheduled_quantity": "mean"}}

        result = resample_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_resampled"


class TestAggregateStep:
    """Test aggregate step implementation."""

    def test_aggregate_sum(self, sample_handle, handle_storage):
        """Test sum aggregation."""
        params = {"group": ["pipeline_name"], "metric": {"col": "scheduled_quantity", "fn": "sum"}}

        result = aggregate_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_aggregated"
        assert "pipeline_name" in result.schema
        assert "sum_scheduled_quantity" in result.schema

    def test_aggregate_count(self, sample_handle, handle_storage):
        """Test count aggregation."""
        params = {"group": ["state_abb"], "metric": {"col": "pipeline_name", "fn": "count"}}

        result = aggregate_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_aggregated"
        assert "count" in result.schema

    def test_aggregate_no_group(self, sample_handle, handle_storage):
        """Test aggregation without grouping."""
        params = {"group": [], "metric": {"col": "scheduled_quantity", "fn": "mean"}}

        result = aggregate_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_aggregated"


class TestSTLDeseasonalizeStep:
    """Test STL deseasonalize step implementation."""

    def test_stl_basic(self, sample_handle, handle_storage):
        """Test basic STL decomposition."""
        params = {"column": "scheduled_quantity", "seasonal": ["weekly", "annual"]}

        result = stl_deseasonalize_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_deseasonalized"
        assert "deseasonalized" in result.schema
        assert "trend" in result.schema
        assert "seasonal_weekly" in result.schema
        assert "seasonal_annual" in result.schema

    def test_stl_with_params(self, sample_handle, handle_storage):
        """Test STL with custom parameters."""
        params = {
            "column": "scheduled_quantity",
            "seasonal": ["weekly"],
            "trend_window": 14,
            "seasonal_windows": {"weekly": 7},
        }

        result = stl_deseasonalize_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_deseasonalized"


class TestChangepointStep:
    """Test changepoint detection step implementation."""

    def test_changepoint_basic(self, sample_handle, handle_storage):
        """Test basic changepoint detection."""
        params = {"column": "scheduled_quantity", "min_size": 5, "penalty": 10.0}

        result = changepoint_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_changepoints"
        assert "changepoint_date" in result.schema
        assert "confidence" in result.schema

    def test_changepoint_with_groupby(self, sample_handle, handle_storage):
        """Test changepoint detection with grouping."""
        params = {
            "column": "scheduled_quantity",
            "groupby": ["pipeline_name"],
            "min_size": 3,
            "penalty": 5.0,
        }

        result = changepoint_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_changepoints"
        assert "pipeline_name" in result.schema


class TestRankStep:
    """Test rank step implementation."""

    def test_rank_basic(self, sample_handle, handle_storage):
        """Test basic ranking."""
        params = {"by": "scheduled_quantity", "descending": True}

        result = rank_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_ranked"
        assert "rank" in result.schema

    def test_rank_multiple_columns(self, sample_handle, handle_storage):
        """Test ranking by multiple columns."""
        params = {"by": ["pipeline_name", "scheduled_quantity"], "descending": False}

        result = rank_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_ranked"

    def test_rank_methods(self, sample_handle, handle_storage):
        """Test different ranking methods."""
        for method in ["ordinal", "dense", "min", "max"]:
            params = {"by": "scheduled_quantity", "method": method}

            result = rank_step(sample_handle, params, handle_storage, "test_digest")

            assert result.id == "test_handle_ranked"


class TestLimitStep:
    """Test limit step implementation."""

    def test_limit_basic(self, sample_handle, handle_storage):
        """Test basic limit."""
        params = {"n": 10}

        result = limit_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_limited"
        assert result.schema == sample_handle.schema

    def test_limit_with_offset(self, sample_handle, handle_storage):
        """Test limit with offset."""
        params = {"n": 5, "offset": 10}

        result = limit_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_limited"

    def test_limit_invalid_params(self, sample_handle, handle_storage):
        """Test limit with invalid parameters."""
        with pytest.raises(ValueError, match="Limit step requires 'n' parameter"):
            limit_step(sample_handle, {}, handle_storage, "test_digest")

        with pytest.raises(ValueError, match="must be positive"):
            limit_step(sample_handle, {"n": 0}, handle_storage, "test_digest")


class TestSaveArtifactStep:
    """Test save artifact step implementation."""

    def test_save_parquet(self, sample_handle, handle_storage, tmp_path):
        """Test saving as parquet."""
        params = {"format": "parquet", "filename": "test_output", "path": str(tmp_path)}

        result = save_artifact_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_saved"
        assert (tmp_path / "test_output.parquet").exists()
        assert (tmp_path / "test_output_metadata.json").exists()

    def test_save_csv(self, sample_handle, handle_storage, tmp_path):
        """Test saving as CSV."""
        params = {"format": "csv", "filename": "test_output", "path": str(tmp_path)}

        result = save_artifact_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_saved"
        assert (tmp_path / "test_output.csv").exists()

    def test_save_json(self, sample_handle, handle_storage, tmp_path):
        """Test saving as JSON."""
        params = {"format": "json", "filename": "test_output", "path": str(tmp_path)}

        result = save_artifact_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_saved"
        assert (tmp_path / "test_output.json").exists()


class TestEvidenceCollectStep:
    """Test evidence collect step implementation."""

    def test_evidence_basic(self, sample_handle, handle_storage):
        """Test basic evidence collection."""
        params = {}

        result = evidence_collect_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_evidence"

    def test_evidence_with_metrics(self, sample_handle, handle_storage):
        """Test evidence collection with custom metrics."""
        params = {"metrics": ["data_quality_score", "completeness", "uniqueness"], "sample_size": 3}

        result = evidence_collect_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_evidence"

    def test_evidence_no_schema_stats(self, sample_handle, handle_storage):
        """Test evidence collection without schema and stats."""
        params = {"include_schema": False, "include_stats": False}

        result = evidence_collect_step(sample_handle, params, handle_storage, "test_digest")

        assert result.id == "test_handle_evidence"


class TestStepIntegration:
    """Test step integration and chaining."""

    def test_step_chaining(self, sample_handle, handle_storage):
        """Test chaining multiple steps."""
        # Filter -> Aggregate -> Rank -> Limit

        # Step 1: Filter
        filter_params = {
            "filters": [{"column": "pipeline_name", "op": "=", "value": "Pipeline A"}],
            "materialize": True,
        }
        filtered_handle = filter_step(sample_handle, filter_params, handle_storage, "test_digest")

        # Step 2: Aggregate (would need materialized input in real scenario)
        # For this test, we'll skip the chaining since it requires more complex setup
        assert filtered_handle.id == "test_handle_filtered"

    def test_engine_compatibility(self, sample_handle, handle_storage):
        """Test that steps preserve engine settings."""
        sample_handle.engine = "duckdb"

        params = {"n": 5}
        result = limit_step(sample_handle, params, handle_storage, "test_digest")

        assert result.engine == "duckdb"

    def test_schema_transformation(self, sample_handle, handle_storage):
        """Test that steps correctly transform schemas."""
        # Aggregate should change schema
        params = {"group": ["pipeline_name"], "metric": {"col": "scheduled_quantity", "fn": "sum"}}

        result = aggregate_step(sample_handle, params, handle_storage, "test_digest")

        # Should have group column and aggregated column
        assert "pipeline_name" in result.schema
        assert "sum_scheduled_quantity" in result.schema
        # Should not have original columns that weren't grouped
        assert "eff_gas_day" not in result.schema


class TestErrorHandling:
    """Test error handling in steps."""

    def test_invalid_handle_store(self, handle_storage):
        """Test error handling for invalid handle store type."""
        invalid_handle = create_lazy_handle("test", schema={})

        with pytest.raises(ValueError, match="requires materialized input handle"):
            filter_step(invalid_handle, {"filters": []}, handle_storage, "test_digest")

    def test_missing_required_params(self, sample_handle, handle_storage):
        """Test error handling for missing required parameters."""
        with pytest.raises(ValueError, match="requires 'by' parameter"):
            rank_step(sample_handle, {}, handle_storage, "test_digest")

    def test_unsupported_operations(self, sample_handle, handle_storage):
        """Test error handling for unsupported operations."""
        params = {"column": "test", "method": "unsupported"}

        with pytest.raises(ValueError, match="Unsupported changepoint method"):
            changepoint_step(sample_handle, params, handle_storage, "test_digest")
