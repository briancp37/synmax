"""Tests for the agent executor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.data_agent.core.agent_executor import AgentExecutor, execute
from src.data_agent.core.agent_schema import (
    Edge,
    PlanGraph,
    Step,
    StepType,
)
from src.data_agent.core.handles import HandleStorage, StepHandle, StepStats


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pl.DataFrame(
        {
            "pipeline_name": ["Pipeline A", "Pipeline B", "Pipeline A", "Pipeline B"],
            "scheduled_quantity": [100.0, 200.0, 150.0, 250.0],
            "eff_gas_day": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "state_abb": ["TX", "LA", "TX", "LA"],
        }
    ).with_columns(pl.col("eff_gas_day").str.to_date())


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield HandleStorage(Path(temp_dir))


@pytest.fixture
def sample_dataset_handle(sample_df, temp_storage):
    """Create a sample dataset handle."""
    # Write sample data to parquet
    temp_path = temp_storage.base_dir / "sample_data.parquet"
    sample_df.write_parquet(temp_path)

    # Create handle
    stats = StepStats(
        rows=sample_df.height,
        bytes=temp_path.stat().st_size,
        columns=sample_df.width,
        null_count={col: sample_df[col].null_count() for col in sample_df.columns},
        computed_at=1234567890.0,
    )

    return StepHandle(
        id="raw",
        store="parquet",
        path=temp_path,
        engine="polars",
        schema={col: str(dtype) for col, dtype in zip(sample_df.columns, sample_df.dtypes)},
        stats=stats,
        fingerprint="sample_fingerprint",
    )


class TestAgentExecutor:
    """Test cases for AgentExecutor."""

    def test_simple_filter_execution(self, sample_dataset_handle, temp_storage):
        """Test executing a simple filter operation."""
        # Create a simple filter plan
        plan = PlanGraph(
            nodes=[
                Step(
                    id="filter_step",
                    op=StepType.FILTER,
                    params={"column": "pipeline_name", "op": "=", "value": "Pipeline A"},
                )
            ],
            edges=[Edge(src="raw", dst="filter_step")],
            inputs=["raw"],
            outputs=["filter_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        # Check results
        assert result_df.height == 2  # Should have 2 rows for Pipeline A
        assert all(result_df["pipeline_name"] == "Pipeline A")

        # Check evidence
        assert evidence["plan"]["plan_hash"] == plan.plan_hash()
        assert len(evidence["steps"]) == 1
        assert evidence["steps"][0]["node_id"] == "filter_step"
        assert "timings" in evidence["steps"][0]
        assert "snippet" in evidence["steps"][0]

    def test_aggregate_execution(self, sample_dataset_handle, temp_storage):
        """Test executing an aggregation operation."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="agg_step",
                    op=StepType.AGGREGATE,
                    params={
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                )
            ],
            edges=[Edge(src="raw", dst="agg_step")],
            inputs=["raw"],
            outputs=["agg_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        # Check results
        assert result_df.height == 2  # Should have 2 groups
        assert "sum_scheduled_quantity" in result_df.columns
        assert set(result_df["pipeline_name"]) == {"Pipeline A", "Pipeline B"}

        # Check evidence
        assert len(evidence["steps"]) == 1
        assert evidence["steps"][0]["node_id"] == "agg_step"

    def test_multi_step_execution(self, sample_dataset_handle, temp_storage):
        """Test executing a multi-step plan."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="filter_step",
                    op=StepType.FILTER,
                    params={"column": "scheduled_quantity", "op": ">", "value": 120.0},
                ),
                Step(
                    id="agg_step",
                    op=StepType.AGGREGATE,
                    params={
                        "groupby": ["state_abb"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                ),
                Step(
                    id="limit_step",
                    op=StepType.LIMIT,
                    params={"n": 1},
                ),
            ],
            edges=[
                Edge(src="raw", dst="filter_step"),
                Edge(src="filter_step", dst="agg_step"),
                Edge(src="agg_step", dst="limit_step"),
            ],
            inputs=["raw"],
            outputs=["limit_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        # Check results
        assert result_df.height == 1  # Limited to 1 row
        assert "sum_scheduled_quantity" in result_df.columns

        # Check evidence - should have 3 steps
        assert len(evidence["steps"]) == 3
        step_ids = [step["node_id"] for step in evidence["steps"]]
        assert step_ids == ["filter_step", "agg_step", "limit_step"]

        # Check timings are present for all steps
        for step in evidence["steps"]:
            assert "timings" in step
            assert "total" in step["timings"]
            assert step["timings"]["total"] > 0

    def test_evidence_collection_step(self, sample_dataset_handle, temp_storage):
        """Test evidence collection step."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="evidence_step",
                    op=StepType.EVIDENCE_COLLECT,
                    params={"sample_size": 2, "method": "head"},
                )
            ],
            edges=[Edge(src="raw", dst="evidence_step")],
            inputs=["raw"],
            outputs=["evidence_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        # Check results
        assert result_df.height == 2  # Should be limited by sample_size

        # Check evidence
        assert len(evidence["steps"]) == 1
        assert evidence["steps"][0]["node_id"] == "evidence_step"

    def test_invalid_step_type(self, sample_dataset_handle, temp_storage):
        """Test handling of invalid step types."""
        # Test that invalid step types are caught by pydantic validation
        with pytest.raises(ValueError):
            Step(
                id="invalid_step",
                op="unsupported_op",  # This will cause validation error
                params={},
            )

    def test_missing_input_handle(self, sample_dataset_handle, temp_storage):
        """Test handling of missing input handles."""
        # Test that invalid edge sources are caught by pydantic validation
        with pytest.raises(ValueError):
            PlanGraph(
                nodes=[
                    Step(
                        id="filter_step",
                        op=StepType.FILTER,
                        params={"column": "pipeline_name", "op": "=", "value": "Pipeline A"},
                    )
                ],
                edges=[Edge(src="nonexistent", dst="filter_step")],  # Invalid source
                inputs=["raw"],
                outputs=["filter_step"],
            )

    def test_checkpoint_decision(self, temp_storage):
        """Test checkpointing decision logic."""
        # Test with large data that should trigger checkpointing
        large_stats = StepStats(
            rows=200_000,  # Above ROW_CKPT threshold
            bytes=200_000_000,  # Above BYTE_CKPT threshold
            columns=10,
            null_count={},
            computed_at=1234567890.0,
        )

        should_checkpoint = temp_storage.should_checkpoint(
            large_stats, "filter", False, 100_000, 100_000_000
        )
        assert should_checkpoint

        # Test with expensive operation
        should_checkpoint = temp_storage.should_checkpoint(
            None, "stl_deseasonalize", False, 100_000, 100_000_000
        )
        assert should_checkpoint

        # Test with multiple consumers
        small_stats = StepStats(
            rows=1000,
            bytes=10_000,
            columns=5,
            null_count={},
            computed_at=1234567890.0,
        )

        should_checkpoint = temp_storage.should_checkpoint(
            small_stats, "filter", True, 100_000, 100_000_000  # has_multiple_consumers=True
        )
        assert should_checkpoint

    def test_code_snippet_generation(self, sample_dataset_handle, temp_storage):
        """Test code snippet generation for different operations."""
        # Test filter snippet
        plan = PlanGraph(
            nodes=[
                Step(
                    id="filter_step",
                    op=StepType.FILTER,
                    params={"column": "pipeline_name", "op": "=", "value": "Pipeline A"},
                )
            ],
            edges=[Edge(src="raw", dst="filter_step")],
            inputs=["raw"],
            outputs=["filter_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        snippet = evidence["steps"][0]["snippet"]
        assert "pl.col('pipeline_name') == 'Pipeline A'" in snippet
        assert "import polars as pl" in snippet

        # Test aggregate snippet
        plan = PlanGraph(
            nodes=[
                Step(
                    id="agg_step",
                    op=StepType.AGGREGATE,
                    params={
                        "groupby": ["state_abb"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                )
            ],
            edges=[Edge(src="raw", dst="agg_step")],
            inputs=["raw"],
            outputs=["agg_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        snippet = evidence["steps"][0]["snippet"]
        assert "group_by" in snippet
        assert "sum_scheduled_quantity" in snippet

    def test_step_statistics_collection(self, sample_dataset_handle, temp_storage):
        """Test that step statistics are properly collected."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="filter_step",
                    op=StepType.FILTER,
                    params={"column": "pipeline_name", "op": "=", "value": "Pipeline A"},
                )
            ],
            edges=[Edge(src="raw", dst="filter_step")],
            inputs=["raw"],
            outputs=["filter_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        step_evidence = evidence["steps"][0]

        # Check input stats
        assert "input_stats" in step_evidence
        assert step_evidence["input_stats"]["handles"] == 1
        assert step_evidence["input_stats"]["total_rows"] > 0

        # Check output stats
        assert "output_stats" in step_evidence
        assert step_evidence["output_stats"]["rows"] == 2  # Filtered to Pipeline A
        assert step_evidence["output_stats"]["columns"] == 4

        # Check timings
        assert "timings" in step_evidence
        timing_keys = {"load", "execute", "collect", "materialize", "total"}
        assert set(step_evidence["timings"].keys()) == timing_keys
        assert all(t >= 0 for t in step_evidence["timings"].values())


class TestExecuteFunction:
    """Test the standalone execute function."""

    def test_execute_function(self, sample_dataset_handle):
        """Test the standalone execute function."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="filter_step",
                    op=StepType.FILTER,
                    params={"column": "pipeline_name", "op": "=", "value": "Pipeline A"},
                )
            ],
            edges=[Edge(src="raw", dst="filter_step")],
            inputs=["raw"],
            outputs=["filter_step"],
        )

        result_df, evidence = execute(plan, sample_dataset_handle)

        # Check that results are returned correctly
        assert isinstance(result_df, pl.DataFrame)
        assert isinstance(evidence, dict)
        assert result_df.height == 2
        assert len(evidence["steps"]) == 1


class TestFilterOperations:
    """Test various filter operations."""

    def test_between_filter(self, sample_dataset_handle, temp_storage):
        """Test between filter operation."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="filter_step",
                    op=StepType.FILTER,
                    params={
                        "column": "scheduled_quantity",
                        "op": "between",
                        "value": [120.0, 200.0],
                    },
                )
            ],
            edges=[Edge(src="raw", dst="filter_step")],
            inputs=["raw"],
            outputs=["filter_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        # Should include rows with scheduled_quantity between 120 and 200
        assert result_df.height == 2
        assert all(
            (result_df["scheduled_quantity"] >= 120) & (result_df["scheduled_quantity"] <= 200)
        )

    def test_in_filter(self, sample_dataset_handle, temp_storage):
        """Test in filter operation."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="filter_step",
                    op=StepType.FILTER,
                    params={
                        "column": "state_abb",
                        "op": "in",
                        "value": ["TX"],
                    },
                )
            ],
            edges=[Edge(src="raw", dst="filter_step")],
            inputs=["raw"],
            outputs=["filter_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        # Should only include TX rows
        assert result_df.height == 2
        assert all(result_df["state_abb"] == "TX")

    def test_is_not_null_filter(self, temp_storage):
        """Test is_not_null filter operation."""
        # Create data with nulls
        df_with_nulls = pl.DataFrame(
            {
                "pipeline_name": ["Pipeline A", None, "Pipeline B", "Pipeline C"],
                "scheduled_quantity": [100.0, 200.0, None, 250.0],
                "state_abb": ["TX", "LA", "TX", "LA"],
            }
        )

        # Write to temp file
        temp_path = temp_storage.base_dir / "null_data.parquet"
        df_with_nulls.write_parquet(temp_path)

        dataset_handle = StepHandle(
            id="raw",
            store="parquet",
            path=temp_path,
            engine="polars",
            schema={
                col: str(dtype) for col, dtype in zip(df_with_nulls.columns, df_with_nulls.dtypes)
            },
            stats=StepStats(
                rows=df_with_nulls.height,
                bytes=temp_path.stat().st_size,
                columns=df_with_nulls.width,
                null_count={col: df_with_nulls[col].null_count() for col in df_with_nulls.columns},
                computed_at=1234567890.0,
            ),
            fingerprint="null_data_fingerprint",
        )

        plan = PlanGraph(
            nodes=[
                Step(
                    id="filter_step",
                    op=StepType.FILTER,
                    params={"column": "pipeline_name", "op": "is_not_null"},
                )
            ],
            edges=[Edge(src="raw", dst="filter_step")],
            inputs=["raw"],
            outputs=["filter_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, dataset_handle)

        # Should exclude the null pipeline_name row
        assert result_df.height == 3
        assert result_df["pipeline_name"].null_count() == 0


class TestAggregateOperations:
    """Test various aggregation operations."""

    def test_multiple_metrics(self, sample_dataset_handle, temp_storage):
        """Test aggregation with multiple metrics."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="agg_step",
                    op=StepType.AGGREGATE,
                    params={
                        "groupby": ["state_abb"],
                        "metrics": [
                            {"col": "scheduled_quantity", "fn": "sum"},
                            {"col": "scheduled_quantity", "fn": "avg"},
                            {"col": "pipeline_name", "fn": "count"},
                        ],
                    },
                )
            ],
            edges=[Edge(src="raw", dst="agg_step")],
            inputs=["raw"],
            outputs=["agg_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        # Check that all metrics are present
        expected_cols = {"state_abb", "sum_scheduled_quantity", "avg_scheduled_quantity", "count"}
        assert set(result_df.columns) == expected_cols
        assert result_df.height == 2  # TX and LA

    def test_no_groupby_aggregation(self, sample_dataset_handle, temp_storage):
        """Test aggregation without groupby (global aggregation)."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="agg_step",
                    op=StepType.AGGREGATE,
                    params={
                        "groupby": [],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                )
            ],
            edges=[Edge(src="raw", dst="agg_step")],
            inputs=["raw"],
            outputs=["agg_step"],
        )

        executor = AgentExecutor(temp_storage)
        result_df, evidence = executor.execute(plan, sample_dataset_handle)

        # Should have single row with global sum
        assert result_df.height == 1
        assert "sum_scheduled_quantity" in result_df.columns
        assert result_df["sum_scheduled_quantity"][0] == 700.0  # Sum of all values
