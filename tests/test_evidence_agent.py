"""Test evidence collection and plan persistence for agent execution."""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.data_agent.core.agent_executor import AgentExecutor, execute
from src.data_agent.core.agent_schema import Edge, PlanGraph, Step, StepType
from src.data_agent.core.evidence import (
    StepEvidence,
    generate_step_code_snippet,
    save_plan_evidence,
)
from src.data_agent.core.handles import HandleStorage


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        "pipeline_name": ["Pipeline_A", "Pipeline_B", "Pipeline_A", "Pipeline_C"] * 25,
        "state_abb": ["TX", "CA", "TX", "NY"] * 25,
        "scheduled_quantity": [100.0, 150.0, 120.0, 80.0] * 25,
        "eff_gas_day": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"] * 25,
    }
    df = pl.DataFrame(data)
    df = df.with_columns(pl.col("eff_gas_day").str.strptime(pl.Date, "%Y-%m-%d"))
    return df


@pytest.fixture
def sample_plan():
    """Create a sample DAG plan for testing."""
    return PlanGraph(
        nodes=[
            Step(
                id="filter_2021",
                op=StepType.FILTER,
                params={
                    "column": "eff_gas_day",
                    "op": "between",
                    "value": ["2021-01-01", "2021-12-31"],
                },
            ),
            Step(
                id="aggregate_by_pipeline",
                op=StepType.AGGREGATE,
                params={
                    "groupby": ["pipeline_name"],
                    "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                },
            ),
            Step(
                id="rank_by_sum",
                op=StepType.RANK,
                params={"by": ["sum_scheduled_quantity"], "descending": True},
            ),
            Step(id="limit_top_5", op=StepType.LIMIT, params={"n": 5}),
            Step(
                id="collect_evidence",
                op=StepType.EVIDENCE_COLLECT,
                params={"sample_size": 10, "method": "head"},
            ),
        ],
        edges=[
            Edge(src="raw", dst="filter_2021"),
            Edge(src="filter_2021", dst="aggregate_by_pipeline"),
            Edge(src="aggregate_by_pipeline", dst="rank_by_sum"),
            Edge(src="rank_by_sum", dst="limit_top_5"),
            Edge(src="limit_top_5", dst="collect_evidence"),
        ],
        inputs=["raw"],
        outputs=["collect_evidence"],
    )


@pytest.fixture
def temp_dataset_handle(sample_dataset):
    """Create a temporary dataset handle."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_data.parquet"
        sample_dataset.write_parquet(temp_path)

        handle_storage = HandleStorage()
        handle = handle_storage.materialize_handle(
            sample_dataset, "test_dataset", "test_dataset", "polars"
        )
        yield handle


class TestStepEvidence:
    """Test step evidence collection."""

    def test_step_evidence_creation(self):
        """Test creating step evidence."""
        evidence = StepEvidence(
            node_id="test_step",
            params={"column": "test", "op": "=", "value": "test_value"},
            input_stats={"rows": 100, "bytes": 1024, "columns": 5},
            output_stats={"rows": 50, "bytes": 512, "columns": 5},
            timings={
                "load": 0.1,
                "execute": 0.2,
                "collect": 0.05,
                "materialize": 0.15,
                "total": 0.5,
            },
            snippet=(
                "import polars as pl\n"
                "lf = pl.scan_parquet('input.parquet')\n"
                "result = lf.filter(pl.col('test') == 'test_value')\n"
                "print(result.collect())"
            ),
            checkpoint_path="/path/to/checkpoint.parquet",
        )

        assert evidence.node_id == "test_step"
        assert evidence.params["column"] == "test"
        assert evidence.input_stats["rows"] == 100
        assert evidence.output_stats["rows"] == 50
        assert evidence.timings["total"] == 0.5
        assert "filter" in evidence.snippet
        assert evidence.checkpoint_path == "/path/to/checkpoint.parquet"


class TestCodeSnippetGeneration:
    """Test code snippet generation for different step types."""

    def test_filter_snippet_equals(self):
        """Test filter snippet generation for equals operation."""
        snippet = generate_step_code_snippet(
            "filter", {"column": "state_abb", "op": "=", "value": "TX"}, "input.parquet"
        )

        assert "import polars as pl" in snippet
        assert 'pl.scan_parquet("input.parquet")' in snippet
        assert "pl.col('state_abb') == 'TX'" in snippet
        assert "print(result.collect())" in snippet

    def test_filter_snippet_between(self):
        """Test filter snippet generation for between operation."""
        snippet = generate_step_code_snippet(
            "filter",
            {"column": "eff_gas_day", "op": "between", "value": ["2021-01-01", "2021-12-31"]},
            "input.parquet",
        )

        assert "pl.col('eff_gas_day') >= '2021-01-01'" in snippet
        assert "pl.col('eff_gas_day') <= '2021-12-31'" in snippet

    def test_aggregate_snippet(self):
        """Test aggregate snippet generation."""
        snippet = generate_step_code_snippet(
            "aggregate",
            {"groupby": ["pipeline_name"], "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]},
        )

        assert "group_by(['pipeline_name'])" in snippet
        assert "pl.col('scheduled_quantity').sum().alias('sum_scheduled_quantity')" in snippet

    def test_rank_snippet(self):
        """Test rank snippet generation."""
        snippet = generate_step_code_snippet(
            "rank", {"by": ["sum_scheduled_quantity"], "method": "average", "descending": True}
        )

        assert "pl.col('sum_scheduled_quantity').rank(method='average', descending=True)" in snippet
        assert "alias('rank_sum_scheduled_quantity')" in snippet

    def test_limit_snippet(self):
        """Test limit snippet generation."""
        snippet = generate_step_code_snippet("limit", {"n": 10, "offset": 0})

        assert "lf.head(10)" in snippet

    def test_limit_snippet_with_offset(self):
        """Test limit snippet with offset."""
        snippet = generate_step_code_snippet("limit", {"n": 10, "offset": 5})

        assert "lf.slice(5, 10)" in snippet

    def test_stl_deseasonalize_snippet(self):
        """Test STL deseasonalize snippet generation."""
        snippet = generate_step_code_snippet(
            "stl_deseasonalize", {"column": "value", "period": 7, "seasonal": 7, "trend": 21}
        )

        assert "STL Deseasonalization on column 'value'" in snippet
        assert "Period: 7, Seasonal: 7, Trend: 21" in snippet
        assert "statsmodels.tsa.seasonal.STL" in snippet

    def test_changepoint_snippet(self):
        """Test changepoint snippet generation."""
        snippet = generate_step_code_snippet(
            "changepoint", {"column": "value", "method": "pelt", "min_size": 5}
        )

        assert "Changepoint detection on column 'value'" in snippet
        assert "Method: pelt, Min size: 5" in snippet
        assert "ruptures library" in snippet


class TestPlanEvidenceSaving:
    """Test saving plan evidence to files."""

    def test_save_plan_evidence(self, sample_plan, sample_dataset):
        """Test saving comprehensive plan evidence."""
        # Create sample step evidence
        step_evidence = [
            StepEvidence(
                node_id="filter_2021",
                params={
                    "column": "eff_gas_day",
                    "op": "between",
                    "value": ["2021-01-01", "2021-12-31"],
                },
                input_stats={"rows": 100, "bytes": 1024, "columns": 4},
                output_stats={"rows": 100, "bytes": 1024, "columns": 4},
                timings={
                    "load": 0.1,
                    "execute": 0.05,
                    "collect": 0.02,
                    "materialize": 0.0,
                    "total": 0.17,
                },
                snippet=(
                    "import polars as pl\n"
                    "# Step: filter\n"
                    "lf = pl.scan_parquet('input.parquet')\n"
                    "result = lf.filter((pl.col('eff_gas_day') >= '2021-01-01') & "
                    "(pl.col('eff_gas_day') <= '2021-12-31'))\n"
                    "print(result.collect())"
                ),
                checkpoint_path=None,
            ),
            StepEvidence(
                node_id="aggregate_by_pipeline",
                params={
                    "groupby": ["pipeline_name"],
                    "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                },
                input_stats={"rows": 100, "bytes": 1024, "columns": 4},
                output_stats={"rows": 4, "bytes": 128, "columns": 2},
                timings={
                    "load": 0.05,
                    "execute": 0.1,
                    "collect": 0.02,
                    "materialize": 0.05,
                    "total": 0.22,
                },
                snippet=(
                    "import polars as pl\n"
                    "# Step: aggregate\n"
                    "lf = pl.scan_parquet('input.parquet')\n"
                    "result = lf.group_by(['pipeline_name']).agg("
                    "[pl.col('scheduled_quantity').sum().alias('sum_scheduled_quantity')])\n"
                    "print(result.collect())"
                ),
                checkpoint_path="/tmp/checkpoint.parquet",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "evidence.json"
            saved_path = save_plan_evidence(
                sample_plan, step_evidence, sample_dataset, 0.39, output_path
            )

            assert saved_path == output_path
            assert output_path.exists()

            # Load and verify the evidence file
            with open(output_path) as f:
                evidence_doc = json.load(f)

            # Check structure
            assert "plan" in evidence_doc
            assert "steps" in evidence_doc
            assert "final_result" in evidence_doc
            assert "metadata" in evidence_doc

            # Check plan section
            plan_section = evidence_doc["plan"]
            assert "plan_hash" in plan_section
            assert "nodes" in plan_section
            assert "edges" in plan_section
            assert len(plan_section["nodes"]) == 5
            assert len(plan_section["edges"]) == 5

            # Check steps section
            steps_section = evidence_doc["steps"]
            assert len(steps_section) == 2
            assert steps_section[0]["node_id"] == "filter_2021"
            assert steps_section[1]["node_id"] == "aggregate_by_pipeline"
            assert steps_section[0]["checkpoint_path"] is None
            assert steps_section[1]["checkpoint_path"] == "/tmp/checkpoint.parquet"

            # Check final result section
            final_result = evidence_doc["final_result"]
            assert final_result["rows"] == sample_dataset.height
            assert final_result["columns"] == sample_dataset.width

            # Check metadata
            metadata = evidence_doc["metadata"]
            assert metadata["total_steps"] == 2
            assert metadata["total_time_seconds"] == 0.39
            assert "created_at" in metadata
            assert "replay_token" in metadata
            assert metadata["replay_token"] == plan_section["plan_hash"]


class TestAgentExecutorEvidence:
    """Test evidence collection in agent executor."""

    def test_executor_evidence_collection(self, sample_plan, temp_dataset_handle):
        """Test that executor collects evidence properly."""
        executor = AgentExecutor()
        final_df, evidence = executor.execute(sample_plan, temp_dataset_handle)

        # Check that evidence is collected
        assert "plan" in evidence
        assert "steps" in evidence
        assert "final_result" in evidence
        assert "evidence_file_path" in evidence

        # Check that evidence file was created
        evidence_file_path = Path(evidence["evidence_file_path"])
        assert evidence_file_path.exists()

        # Verify evidence file content
        with open(evidence_file_path) as f:
            evidence_doc = json.load(f)

        assert evidence_doc["plan"]["plan_hash"] == sample_plan.plan_hash()
        assert len(evidence_doc["steps"]) == 5  # All 5 steps should have evidence

        # Check that each step has required evidence fields
        for step_evidence in evidence_doc["steps"]:
            assert "node_id" in step_evidence
            assert "params" in step_evidence
            assert "input_stats" in step_evidence
            assert "output_stats" in step_evidence
            assert "timings" in step_evidence
            assert "snippet" in step_evidence

            # Check that snippet is valid Python code
            snippet = step_evidence["snippet"]
            assert "import polars as pl" in snippet
            assert "print(result.collect())" in snippet

    def test_execute_function_evidence_saving(self, sample_plan, temp_dataset_handle):
        """Test the standalone execute function saves evidence."""
        final_df, evidence = execute(sample_plan, temp_dataset_handle)

        # Check evidence file was created
        assert "evidence_file_path" in evidence
        evidence_file_path = Path(evidence["evidence_file_path"])
        assert evidence_file_path.exists()

        # Verify the evidence file contains plan hash
        with open(evidence_file_path) as f:
            evidence_doc = json.load(f)

        assert evidence_doc["metadata"]["replay_token"] == sample_plan.plan_hash()


class TestEvidenceReproducibility:
    """Test evidence includes reproducibility information."""

    def test_evidence_includes_checkpoint_paths(self, temp_dataset_handle):
        """Test that evidence includes checkpoint paths for materialized steps."""
        # Create a plan that will trigger materialization
        plan = PlanGraph(
            nodes=[
                Step(
                    id="aggregate_heavy",
                    op=StepType.AGGREGATE,
                    params={
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                    materialize=True,  # Force materialization
                ),
                Step(id="limit_result", op=StepType.LIMIT, params={"n": 3}),
            ],
            edges=[
                Edge(src="raw", dst="aggregate_heavy"),
                Edge(src="aggregate_heavy", dst="limit_result"),
            ],
            inputs=["raw"],
            outputs=["limit_result"],
        )

        executor = AgentExecutor()
        final_df, evidence = executor.execute(plan, temp_dataset_handle)

        # Load evidence file
        with open(evidence["evidence_file_path"]) as f:
            evidence_doc = json.load(f)

        # Check that materialized step has checkpoint path
        aggregate_step = next(
            step for step in evidence_doc["steps"] if step["node_id"] == "aggregate_heavy"
        )
        assert aggregate_step["checkpoint_path"] is not None
        assert Path(aggregate_step["checkpoint_path"]).exists()

    def test_evidence_snippets_are_executable(self, sample_plan, temp_dataset_handle):
        """Test that generated code snippets are syntactically valid."""
        executor = AgentExecutor()
        final_df, evidence = executor.execute(sample_plan, temp_dataset_handle)

        # Load evidence file
        with open(evidence["evidence_file_path"]) as f:
            evidence_doc = json.load(f)

        # Check that all snippets are valid Python syntax
        for step_evidence in evidence_doc["steps"]:
            snippet = step_evidence["snippet"]
            try:
                compile(snippet, "<string>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Invalid syntax in snippet for {step_evidence['node_id']}: {e}")


@pytest.mark.integration
class TestEvidenceIntegration:
    """Integration tests for evidence system."""

    def test_evidence_snapshot_consistency(self, sample_plan, temp_dataset_handle):
        """Test that evidence is consistent across multiple runs."""
        # Run the same plan twice
        executor1 = AgentExecutor()
        final_df1, evidence1 = executor1.execute(sample_plan, temp_dataset_handle)

        executor2 = AgentExecutor()
        final_df2, evidence2 = executor2.execute(sample_plan, temp_dataset_handle)

        # Load both evidence files
        with open(evidence1["evidence_file_path"]) as f:
            evidence_doc1 = json.load(f)

        with open(evidence2["evidence_file_path"]) as f:
            evidence_doc2 = json.load(f)

        # Plan hashes should be identical
        assert evidence_doc1["plan"]["plan_hash"] == evidence_doc2["plan"]["plan_hash"]

        # Step counts should be identical
        assert len(evidence_doc1["steps"]) == len(evidence_doc2["steps"])

        # Final results should be identical
        assert evidence_doc1["final_result"]["rows"] == evidence_doc2["final_result"]["rows"]
        assert evidence_doc1["final_result"]["columns"] == evidence_doc2["final_result"]["columns"]

    def test_evidence_file_cleanup(self, sample_plan, temp_dataset_handle):
        """Test evidence files are created in correct location."""
        from src.data_agent.config import ARTIFACTS_DIR

        executor = AgentExecutor()
        final_df, evidence = executor.execute(sample_plan, temp_dataset_handle)

        evidence_file_path = Path(evidence["evidence_file_path"])

        # Check file is in correct directory
        assert evidence_file_path.parent == ARTIFACTS_DIR / "outputs"

        # Check filename format
        assert evidence_file_path.name.endswith(".json")
        assert len(evidence_file_path.stem) == 64  # SHA256 hash length
