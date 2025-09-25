"""Test the regime shifts 2021 worked example end-to-end."""

import json
import os
from pathlib import Path

import polars as pl
import pytest

from data_agent.config import DATA_PATH, PROJECT_ROOT
from data_agent.core.agent_executor import AgentExecutor
from data_agent.core.agent_schema import PlanGraph
from data_agent.core.handles import HandleStorage, StepHandle, StepStats


class TestRegimeShifts2021:
    """Test suite for the regime shifts 2021 worked example."""

    @pytest.fixture
    def plan_path(self) -> Path:
        """Path to the regime shifts 2021 plan JSON."""
        return PROJECT_ROOT / "examples" / "agent_plans" / "regime_shifts_2021.json"

    @pytest.fixture
    def plan_graph(self, plan_path: Path) -> PlanGraph:
        """Load and validate the regime shifts plan."""
        with open(plan_path) as f:
            plan_data = json.load(f)
        return PlanGraph(**plan_data)

    @pytest.fixture
    def dataset_handle(self):
        """Create a handle to the test dataset."""
        if not DATA_PATH.exists():
            pytest.skip(f"Dataset not found at {DATA_PATH}")
        
        # Load dataset to get schema and stats
        import polars as pl
        import time
        
        lf = pl.scan_parquet(DATA_PATH)
        schema = lf.collect_schema()
        stats = StepStats(
            rows=lf.select(pl.len()).collect().item(),
            bytes=DATA_PATH.stat().st_size,
            columns=len(schema),
            null_count={},
            computed_at=time.time(),
        )
        
        return StepHandle(
            id="raw",
            store="parquet",
            path=DATA_PATH,
            engine="polars",
            schema={col: str(dtype) for col, dtype in schema.items()},
            stats=stats,
            fingerprint="dataset",
        )

    @pytest.fixture
    def executor(self):
        """Create an agent executor."""
        return AgentExecutor()

    def test_plan_loads_and_validates(self, plan_graph: PlanGraph):
        """Test that the plan JSON loads and validates correctly."""
        # Basic structure validation
        assert len(plan_graph.nodes) == 7
        assert len(plan_graph.edges) == 7
        assert plan_graph.inputs == ["raw"]
        assert plan_graph.outputs == ["l"]

        # Check step IDs and operations
        step_ops = {step.id: step.op for step in plan_graph.nodes}
        expected_ops = {
            "f": "filter",
            "a": "aggregate", 
            "s": "stl_deseasonalize",
            "c": "changepoint",
            "r": "rank",
            "l": "limit",
            "e": "evidence_collect"
        }
        assert step_ops == expected_ops

        # Validate topological order
        topo_order = plan_graph.topological_order()
        assert "f" in topo_order
        assert "a" in topo_order
        assert topo_order.index("f") < topo_order.index("a")  # f comes before a
        assert topo_order.index("a") < topo_order.index("s")  # a comes before s

    def test_plan_parameters(self, plan_graph: PlanGraph):
        """Test that plan parameters are correctly specified."""
        steps_by_id = {step.id: step for step in plan_graph.nodes}
        
        # Filter step should filter for 2022 (dataset starts from 2022)
        filter_step = steps_by_id["f"]
        assert filter_step.params["column"] == "eff_gas_day"
        assert filter_step.params["op"] == "between"
        assert filter_step.params["value"] == ["2022-01-01", "2022-12-31"]

        # Aggregate step should group by pipeline_name and eff_gas_day, sum scheduled_quantity
        agg_step = steps_by_id["a"]
        assert agg_step.params["groupby"] == ["pipeline_name", "eff_gas_day"]
        assert len(agg_step.params["metrics"]) == 1
        assert agg_step.params["metrics"][0]["col"] == "scheduled_quantity"
        assert agg_step.params["metrics"][0]["fn"] == "sum"

        # STL step should deseasonalize the aggregated column
        stl_step = steps_by_id["s"]
        assert stl_step.params["column"] == "sum_scheduled_quantity"

        # Changepoint step should use PELT with min_size=7
        cp_step = steps_by_id["c"]
        assert cp_step.params["method"] == "pelt"
        assert cp_step.params["min_size"] == 7

        # Rank step should rank by change_magnitude descending
        rank_step = steps_by_id["r"]
        assert rank_step.params["by"] == ["change_magnitude"]
        assert rank_step.params["descending"] is True

        # Limit step should limit to 10 rows
        limit_step = steps_by_id["l"]
        assert limit_step.params["n"] == 10

    def test_plan_hash_stability(self, plan_graph: PlanGraph):
        """Test that plan hash is stable for caching."""
        hash1 = plan_graph.plan_hash()
        hash2 = plan_graph.plan_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="Dataset not available")
    def test_end_to_end_execution(self, plan_graph: PlanGraph, dataset_handle, executor: AgentExecutor):
        """Test end-to-end execution of the regime shifts plan."""
        # Execute the plan
        result_df, evidence = executor.execute(plan_graph, dataset_handle)

        # Verify result structure
        assert isinstance(result_df, pl.DataFrame)
        assert result_df.height <= 10  # Should be limited to 10 rows
        assert result_df.height > 0  # Should have some results

        # Verify evidence structure
        assert "plan" in evidence
        assert "steps" in evidence
        assert "final_result" in evidence
        assert "total_steps" in evidence
        assert "total_time" in evidence

        # Verify plan evidence
        plan_evidence = evidence["plan"]
        assert plan_evidence["plan_hash"] == plan_graph.plan_hash()
        assert len(plan_evidence["nodes"]) == 7
        assert len(plan_evidence["edges"]) == 7

        # Verify step evidence
        step_evidence = evidence["steps"]
        assert len(step_evidence) == 7  # Should have evidence for all 7 steps
        
        # Check that each step has required evidence fields
        for step_ev in step_evidence:
            assert "node_id" in step_ev
            assert "params" in step_ev
            assert "input_stats" in step_ev
            assert "output_stats" in step_ev
            assert "timings" in step_ev
            assert "snippet" in step_ev

        # Verify final result evidence
        final_evidence = evidence["final_result"]
        assert final_evidence["rows"] == result_df.height
        assert final_evidence["columns"] == result_df.width
        assert final_evidence["column_names"] == result_df.columns

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="Dataset not available")
    def test_execution_produces_expected_columns(self, plan_graph: PlanGraph, dataset_handle, executor: AgentExecutor):
        """Test that execution produces expected output columns."""
        result_df, evidence = executor.execute(plan_graph, dataset_handle)

        # Should have pipeline_name from groupby
        assert "pipeline_name" in result_df.columns
        
        # Should have changepoint detection columns
        expected_cp_cols = ["changepoint_date", "change_magnitude", "confidence"]
        for col in expected_cp_cols:
            assert col in result_df.columns, f"Expected column '{col}' not found in {result_df.columns}"

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="Dataset not available") 
    def test_caching_behavior(self, plan_graph: PlanGraph, dataset_handle):
        """Test that plan execution can be cached."""
        executor1 = AgentExecutor()
        executor2 = AgentExecutor()

        # First execution
        result1, evidence1 = executor1.execute(plan_graph, dataset_handle)
        
        # Second execution (should potentially use cached results)
        result2, evidence2 = executor2.execute(plan_graph, dataset_handle)

        # Results should be equivalent
        assert result1.shape == result2.shape
        assert result1.columns == result2.columns

        # Plan hashes should be identical
        assert evidence1["plan"]["plan_hash"] == evidence2["plan"]["plan_hash"]

    def test_plan_json_schema_compliance(self, plan_path: Path):
        """Test that the plan JSON follows the expected schema."""
        with open(plan_path) as f:
            plan_data = json.load(f)

        # Required top-level fields
        assert "nodes" in plan_data
        assert "edges" in plan_data
        assert "inputs" in plan_data
        assert "outputs" in plan_data

        # Node structure validation
        for node in plan_data["nodes"]:
            assert "id" in node
            assert "op" in node
            assert "params" in node
            assert isinstance(node["params"], dict)

        # Edge structure validation
        for edge in plan_data["edges"]:
            assert "src" in edge
            assert "dst" in edge

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="Dataset not available")
    def test_step_execution_order(self, plan_graph: PlanGraph, dataset_handle, executor: AgentExecutor):
        """Test that steps execute in the correct topological order."""
        result_df, evidence = executor.execute(plan_graph, dataset_handle)

        # Extract step execution order from evidence
        step_evidence = evidence["steps"]
        execution_order = [step["node_id"] for step in step_evidence]

        # Verify expected order based on the DAG
        expected_order = ["f", "a", "s", "c", "r", "l", "e"]
        assert execution_order == expected_order

    def test_fallback_behavior_without_llm(self, plan_path: Path):
        """Test that the plan can be loaded directly without LLM."""
        # This tests the --plan file.json functionality
        with open(plan_path) as f:
            plan_data = json.load(f)
        
        # Should be able to create PlanGraph directly
        plan = PlanGraph(**plan_data)
        assert plan is not None
        assert len(plan.nodes) == 7

    @pytest.mark.skipif(
        not DATA_PATH.exists() or not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"), 
        reason="Dataset or API keys not available"
    )
    def test_output_snapshot_format(self, plan_graph: PlanGraph, dataset_handle, executor: AgentExecutor):
        """Test that output matches expected snapshot format."""
        result_df, evidence = executor.execute(plan_graph, dataset_handle)

        # Test result format
        assert result_df.height <= 10  # Limited to 10 rows as specified
        assert result_df.height > 0    # Should have some results

        # Test evidence JSON structure for snapshotting
        assert isinstance(evidence, dict)
        
        # Should be JSON serializable
        json_str = json.dumps(evidence, default=str)  # Use default=str for any non-serializable objects
        assert len(json_str) > 0

        # Evidence should contain plan and execution details
        required_keys = ["plan", "steps", "final_result", "total_steps", "total_time"]
        for key in required_keys:
            assert key in evidence, f"Missing required evidence key: {key}"
