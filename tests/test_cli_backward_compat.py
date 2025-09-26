"""Test backward compatibility for legacy CLI commands.

This module tests that legacy CLI command shapes still work correctly
by compiling them into tiny DAGs and executing through the agent executor.
"""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from data_agent.core.agent_executor import execute as agent_execute
from data_agent.core.agent_shim import (
    build_tiny_dag_from_legacy_args,
    extract_date_range_from_query,
)
from data_agent.core.handles import StepHandle, StepStats


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        "pipeline_name": ["Pipeline_A", "Pipeline_B", "Pipeline_A", "Pipeline_B"] * 25,
        "state_abb": ["TX", "CA", "TX", "NY"] * 25,
        "county_name": ["Harris", "Los Angeles", "Dallas", "New York"] * 25,
        "category_short": ["LDC", "Industrial", "LDC", "Interconnect"] * 25,
        "eff_gas_day": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"] * 25,
        "scheduled_quantity": [100.0, 200.0, 150.0, 300.0] * 25,
    }
    df = pl.DataFrame(data)
    df = df.with_columns(pl.col("eff_gas_day").str.to_date())
    return df


@pytest.fixture
def sample_dataset_handle(sample_dataset):
    """Create a dataset handle for testing."""
    # Write to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        sample_dataset.write_parquet(tmp.name)

        stats = StepStats(
            rows=sample_dataset.height,
            bytes=Path(tmp.name).stat().st_size,
            columns=sample_dataset.width,
            null_count={},
            computed_at=0.0,
        )

        return StepHandle(
            id="raw",
            store="parquet",
            path=Path(tmp.name),
            engine="polars",
            schema={col: str(dtype) for col, dtype in sample_dataset.schema.items()},
            stats=stats,
            fingerprint="test_dataset",
        )


class TestAgentShim:
    """Test the agent shim functionality."""

    def test_build_tiny_dag_top_states(self):
        """Test building DAG for 'top states' query."""
        query = "top 5 states by scheduled quantity in 2022"
        opts = {"date_range": ("2022-01-01", "2022-12-31"), "top": 5}

        plan = build_tiny_dag_from_legacy_args(query, opts)

        # Verify plan structure
        assert len(plan.nodes) == 5  # filter, aggregate, rank, limit, evidence_collect
        assert len(plan.edges) == 5  # raw->f, f->a, a->r, r->l, l->e
        assert plan.inputs == ["raw"]
        assert len(plan.outputs) == 1

        # Verify operations
        node_ops = [node.op for node in plan.nodes]
        assert "filter" in node_ops
        assert "aggregate" in node_ops
        assert "rank" in node_ops
        assert "limit" in node_ops
        assert "evidence_collect" in node_ops

        # Verify filter parameters
        filter_node = next(n for n in plan.nodes if n.op == "filter")
        assert filter_node.params["column"] == "eff_gas_day"
        assert filter_node.params["op"] == "between"
        assert filter_node.params["value"] == ["2022-01-01", "2022-12-31"]

        # Verify aggregate parameters
        agg_node = next(n for n in plan.nodes if n.op == "aggregate")
        assert agg_node.params["groupby"] == ["state_abb"]
        assert len(agg_node.params["metrics"]) == 1
        assert agg_node.params["metrics"][0]["col"] == "scheduled_quantity"
        assert agg_node.params["metrics"][0]["fn"] == "sum"

        # Verify limit parameters
        limit_node = next(n for n in plan.nodes if n.op == "limit")
        assert limit_node.params["n"] == 5

    def test_build_tiny_dag_top_pipelines(self):
        """Test building DAG for 'top pipelines' query."""
        query = "top 10 pipelines by volume"
        opts = {}

        plan = build_tiny_dag_from_legacy_args(query, opts)

        # Verify aggregate groups by pipeline
        agg_node = next(n for n in plan.nodes if n.op == "aggregate")
        assert agg_node.params["groupby"] == ["pipeline_name"]

        # Verify limit is 10
        limit_node = next(n for n in plan.nodes if n.op == "limit")
        assert limit_node.params["n"] == 10

    def test_build_tiny_dag_daily_totals(self):
        """Test building DAG for daily totals query."""
        query = "daily totals by pipeline"
        opts = {"date_range": ("2022-01-01", "2022-12-31")}

        plan = build_tiny_dag_from_legacy_args(query, opts)

        # Should have resample operation for daily aggregation
        node_ops = [node.op for node in plan.nodes]
        assert "resample" in node_ops or "aggregate" in node_ops

    def test_build_tiny_dag_count_query(self):
        """Test building DAG for count-based query."""
        query = "count of pipelines by state"
        opts = {}

        plan = build_tiny_dag_from_legacy_args(query, opts)

        # Verify aggregate uses count
        agg_node = next(n for n in plan.nodes if n.op == "aggregate")
        assert agg_node.params["groupby"] == ["state_abb"]
        assert agg_node.params["metrics"][0]["fn"] == "count"

    def test_build_tiny_dag_fallback(self):
        """Test fallback DAG for unrecognized queries."""
        query = "some random query that doesn't match patterns"
        opts = {}

        plan = build_tiny_dag_from_legacy_args(query, opts)

        # Should create minimal evidence-only plan
        assert len(plan.nodes) == 1
        assert plan.nodes[0].op == "evidence_collect"

    def test_extract_date_range_from_query(self):
        """Test date range extraction from queries."""
        # Test year extraction
        assert extract_date_range_from_query("top states in 2022") == ("2022-01-01", "2022-12-31")
        assert extract_date_range_from_query("data for 2023") == ("2023-01-01", "2023-12-31")

        # Test since pattern
        assert extract_date_range_from_query("since 2022-06-01") == ("2022-06-01", "2030-01-01")

        # Test range pattern
        assert extract_date_range_from_query("from 2022-01-01 to 2022-12-31") == (
            "2022-01-01",
            "2022-12-31",
        )

        # Test no date
        assert extract_date_range_from_query("top states by volume") is None


class TestBackwardCompatibility:
    """Test backward compatibility of legacy commands through full execution."""

    def test_legacy_top_states_execution(self, sample_dataset_handle):
        """Test full execution of 'top states' legacy query."""
        query = "top 3 states by scheduled quantity"
        opts = {}

        # Build DAG
        plan = build_tiny_dag_from_legacy_args(query, opts)

        # Execute DAG
        result_df, evidence = agent_execute(plan, sample_dataset_handle)

        # Verify results
        assert result_df.height <= 3  # Limited to top 3
        assert "state_abb" in result_df.columns
        assert "sum_scheduled_quantity" in result_df.columns

        # Verify evidence
        assert "plan" in evidence
        assert "steps" in evidence
        assert evidence["plan"]["plan_hash"] == plan.plan_hash()
        assert len(evidence["steps"]) > 0

    def test_legacy_pipeline_count_execution(self, sample_dataset_handle):
        """Test full execution of pipeline count query."""
        query = "count of records by pipeline"
        opts = {}

        # Build DAG
        plan = build_tiny_dag_from_legacy_args(query, opts)

        # Execute DAG
        result_df, evidence = agent_execute(plan, sample_dataset_handle)

        # Verify results
        assert "pipeline_name" in result_df.columns
        assert "count" in result_df.columns
        assert result_df.height > 0

        # Verify counts are reasonable
        total_count = result_df["count"].sum()
        assert total_count == 100  # Our sample dataset has 100 rows

    def test_legacy_with_date_filter_execution(self, sample_dataset_handle):
        """Test execution with date filtering."""
        query = "top states by volume in 2022"
        opts = {"date_range": ("2022-01-01", "2022-12-31")}

        # Build DAG
        plan = build_tiny_dag_from_legacy_args(query, opts)

        # Execute DAG
        result_df, evidence = agent_execute(plan, sample_dataset_handle)

        # Should have results (our sample data is in 2022)
        assert result_df.height > 0
        assert "state_abb" in result_df.columns

        # Verify filter was applied in evidence - check node_id for filter steps
        filter_steps = [s for s in evidence["steps"] if s.get("node_id") == "f"]
        assert len(filter_steps) > 0

    def test_plan_hash_consistency(self):
        """Test that identical queries produce identical plan hashes."""
        query = "top 5 states by volume"
        opts = {}

        plan1 = build_tiny_dag_from_legacy_args(query, opts)
        plan2 = build_tiny_dag_from_legacy_args(query, opts)

        assert plan1.plan_hash() == plan2.plan_hash()

    def test_plan_hash_different_for_different_queries(self):
        """Test that different queries produce different plan hashes."""
        plan1 = build_tiny_dag_from_legacy_args("top 5 states by volume", {})
        plan2 = build_tiny_dag_from_legacy_args("top 10 pipelines by volume", {})

        assert plan1.plan_hash() != plan2.plan_hash()


class TestDryRunOutput:
    """Test dry-run output format."""

    def test_dry_run_plan_structure(self):
        """Test that dry-run shows proper plan structure."""
        query = "top 5 states by total scheduled quantity in 2022"
        opts = {"date_range": ("2022-01-01", "2022-12-31")}

        plan = build_tiny_dag_from_legacy_args(query, opts)

        # Test plan serialization (what would be shown in dry-run)
        plan_dict = plan.model_dump()
        plan_json = json.dumps(plan_dict, indent=2)

        # Verify JSON is valid and contains expected structure
        parsed = json.loads(plan_json)
        assert "nodes" in parsed
        assert "edges" in parsed
        assert "inputs" in parsed
        assert "outputs" in parsed

        # Verify topological order can be computed
        topo_order = plan.topological_order()
        assert len(topo_order) > 0
        assert "raw" in topo_order

    def test_plan_estimation(self):
        """Test plan complexity estimation for dry-run."""
        from data_agent.core.agent_planner import estimate_plan_complexity

        query = "top 5 states by volume"
        opts = {}

        plan = build_tiny_dag_from_legacy_args(query, opts)
        estimates = estimate_plan_complexity(plan)

        # Verify estimation structure
        assert "steps" in estimates
        assert "estimated_time_seconds" in estimates
        assert "estimated_memory_mb" in estimates
        assert "topological_order" in estimates

        # Verify reasonable estimates
        assert estimates["steps"] == len(plan.nodes)
        assert estimates["estimated_time_seconds"] > 0
        assert estimates["estimated_memory_mb"] > 0
