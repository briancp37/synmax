"""Tests for deterministic fallbacks and DSL loader functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from data_agent.core.agent_planner import AgentPlanner, PlanValidationError
from data_agent.core.agent_schema import PlanGraph, StepType
from data_agent.core.dsl_loader import (
    DSLLoader,
    DSLValidationError,
    create_macro_plan,
    get_available_macros,
)


class TestDSLLoader:
    """Test DSL loader functionality."""

    def test_load_valid_plan(self):
        """Test loading a valid plan from JSON."""
        plan_data = {
            "nodes": [
                {
                    "id": "a",
                    "op": "aggregate",
                    "params": {
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                },
                {"id": "l", "op": "limit", "params": {"n": 10}},
                {"id": "e", "op": "evidence_collect", "params": {}},
            ],
            "edges": [
                {"src": "raw", "dst": "a"},
                {"src": "a", "dst": "l"},
                {"src": "l", "dst": "e"},
            ],
            "inputs": ["raw"],
            "outputs": ["l"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(plan_data, f)
            temp_path = f.name

        try:
            loader = DSLLoader()
            plan = loader.load_plan(temp_path)

            assert isinstance(plan, PlanGraph)
            assert len(plan.nodes) == 3
            assert len(plan.edges) == 3
            assert plan.inputs == ["raw"]
            assert plan.outputs == ["l"]
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json ")
            temp_path = f.name

        try:
            loader = DSLLoader()
            with pytest.raises(DSLValidationError, match="Invalid JSON"):
                loader.load_plan(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = DSLLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_plan("nonexistent_file.json")

    def test_validate_duplicate_step_ids(self):
        """Test validation catches duplicate step IDs."""
        plan_data = {
            "nodes": [
                {
                    "id": "a",
                    "op": "aggregate",
                    "params": {
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                },
                {"id": "a", "op": "limit", "params": {"n": 10}},  # Duplicate ID
            ],
            "edges": [{"src": "raw", "dst": "a"}],
            "inputs": ["raw"],
            "outputs": ["a"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(plan_data, f)
            temp_path = f.name

        try:
            loader = DSLLoader()
            with pytest.raises(DSLValidationError, match="Duplicate step IDs"):
                loader.load_plan(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_invalid_edge_reference(self):
        """Test validation catches invalid edge references."""
        plan_data = {
            "nodes": [
                {
                    "id": "a",
                    "op": "aggregate",
                    "params": {
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                }
            ],
            "edges": [{"src": "raw", "dst": "nonexistent"}],  # Invalid destination
            "inputs": ["raw"],
            "outputs": ["a"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(plan_data, f)
            temp_path = f.name

        try:
            loader = DSLLoader()
            with pytest.raises(DSLValidationError, match="Invalid plan structure"):
                loader.load_plan(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_cycle_detection(self):
        """Test validation detects cycles."""
        plan_data = {
            "nodes": [
                {
                    "id": "a",
                    "op": "aggregate",
                    "params": {
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                },
                {"id": "b", "op": "limit", "params": {"n": 10}},
            ],
            "edges": [{"src": "a", "dst": "b"}, {"src": "b", "dst": "a"}],  # Creates cycle
            "inputs": ["raw"],
            "outputs": ["b"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(plan_data, f)
            temp_path = f.name

        try:
            loader = DSLLoader()
            with pytest.raises(DSLValidationError, match="Invalid plan structure"):
                loader.load_plan(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_filter_params(self):
        """Test validation of filter step parameters."""
        plan_data = {
            "nodes": [
                {"id": "f", "op": "filter", "params": {"column": "test_col"}}  # Missing 'op' param
            ],
            "edges": [{"src": "raw", "dst": "f"}],
            "inputs": ["raw"],
            "outputs": ["f"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(plan_data, f)
            temp_path = f.name

        try:
            loader = DSLLoader()
            with pytest.raises(DSLValidationError, match="Invalid plan structure"):
                loader.load_plan(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_aggregate_params(self):
        """Test validation of aggregate step parameters."""
        plan_data = {
            "nodes": [
                {
                    "id": "a",
                    "op": "aggregate",
                    "params": {"groupby": ["pipeline_name"]},
                }  # Missing metrics
            ],
            "edges": [{"src": "raw", "dst": "a"}],
            "inputs": ["raw"],
            "outputs": ["a"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(plan_data, f)
            temp_path = f.name

        try:
            loader = DSLLoader()
            with pytest.raises(DSLValidationError, match="Invalid plan structure"):
                loader.load_plan(temp_path)
        finally:
            Path(temp_path).unlink()


class TestMacros:
    """Test macro creation functionality."""

    def test_get_available_macros(self):
        """Test getting list of available macros."""
        macros = get_available_macros()

        assert isinstance(macros, dict)
        assert "simple_aggregation" in macros
        assert "time_series_analysis" in macros
        assert "top_k_ranking" in macros

    def test_create_simple_aggregation_macro(self):
        """Test creating simple aggregation macro."""
        plan = create_macro_plan("simple_aggregation")

        assert isinstance(plan, PlanGraph)
        assert len(plan.nodes) == 3
        assert any(step.op == StepType.AGGREGATE for step in plan.nodes)
        assert any(step.op == StepType.LIMIT for step in plan.nodes)
        assert any(step.op == StepType.EVIDENCE_COLLECT for step in plan.nodes)

    def test_create_simple_aggregation_macro_with_params(self):
        """Test creating simple aggregation macro with custom parameters."""
        plan = create_macro_plan(
            "simple_aggregation",
            groupby_cols=["state_abb"],
            metric_col="scheduled_quantity",
            metric_fn="avg",
            limit=50,
        )

        assert isinstance(plan, PlanGraph)

        # Find the aggregate step and check its parameters
        agg_step = next(step for step in plan.nodes if step.op == StepType.AGGREGATE)
        assert agg_step.params["groupby"] == ["state_abb"]
        assert agg_step.params["metrics"][0]["col"] == "scheduled_quantity"
        assert agg_step.params["metrics"][0]["fn"] == "avg"

        # Find the limit step and check its parameter
        limit_step = next(step for step in plan.nodes if step.op == StepType.LIMIT)
        assert limit_step.params["n"] == 50

    def test_create_time_series_analysis_macro(self):
        """Test creating time series analysis macro."""
        plan = create_macro_plan("time_series_analysis")

        assert isinstance(plan, PlanGraph)
        assert (
            len(plan.nodes) >= 6
        )  # Should have filter, aggregate, stl, changepoint, rank, limit, evidence
        assert any(step.op == StepType.FILTER for step in plan.nodes)
        assert any(step.op == StepType.AGGREGATE for step in plan.nodes)
        assert any(step.op == StepType.STL_DESEASONALIZE for step in plan.nodes)
        assert any(step.op == StepType.CHANGEPOINT for step in plan.nodes)
        assert any(step.op == StepType.RANK for step in plan.nodes)

    def test_create_top_k_ranking_macro(self):
        """Test creating top-k ranking macro."""
        plan = create_macro_plan("top_k_ranking", k=5)

        assert isinstance(plan, PlanGraph)

        # Find the limit step and check k parameter
        limit_step = next(step for step in plan.nodes if step.op == StepType.LIMIT)
        assert limit_step.params["n"] == 5

    def test_create_unknown_macro(self):
        """Test creating unknown macro raises error."""
        with pytest.raises(ValueError, match="Unknown macro"):
            create_macro_plan("nonexistent_macro")


class TestAgentPlannerFallbacks:
    """Test agent planner fallback functionality."""

    def test_fallback_time_series_pattern(self):
        """Test fallback for time series patterns."""
        planner = AgentPlanner(client=None)  # No LLM client
        available_columns = ["pipeline_name", "eff_gas_day", "scheduled_quantity"]

        plan = planner.plan("detect regime shifts in the data", available_columns, fallback=True)

        assert isinstance(plan, PlanGraph)
        assert any(step.op == StepType.STL_DESEASONALIZE for step in plan.nodes)
        assert any(step.op == StepType.CHANGEPOINT for step in plan.nodes)

    def test_fallback_top_k_pattern(self):
        """Test fallback for top-k patterns."""
        planner = AgentPlanner(client=None)  # No LLM client
        available_columns = ["pipeline_name", "scheduled_quantity"]

        plan = planner.plan("show me the top 5 pipelines", available_columns, fallback=True)

        assert isinstance(plan, PlanGraph)
        assert any(step.op == StepType.RANK for step in plan.nodes)

        # Check that limit is set to 5 (extracted from query)
        limit_step = next(step for step in plan.nodes if step.op == StepType.LIMIT)
        assert limit_step.params["n"] == 5

    def test_fallback_aggregation_pattern(self):
        """Test fallback for aggregation patterns."""
        planner = AgentPlanner(client=None)  # No LLM client
        available_columns = ["pipeline_name", "scheduled_quantity"]

        plan = planner.plan("sum the quantities by pipeline", available_columns, fallback=True)

        assert isinstance(plan, PlanGraph)
        assert any(step.op == StepType.AGGREGATE for step in plan.nodes)

    def test_fallback_disabled_raises_error(self):
        """Test that disabling fallback raises error when LLM fails."""
        planner = AgentPlanner(client=None)  # No LLM client
        available_columns = ["pipeline_name", "scheduled_quantity"]

        # When client is None and fallback is False, it should raise an error
        with pytest.raises((PlanValidationError, AttributeError)):
            planner.plan("analyze the data", available_columns, fallback=False)  # Fallback disabled

    def test_plan_from_macro_method(self):
        """Test the plan_from_macro method."""
        planner = AgentPlanner()

        plan = planner.plan_from_macro("simple_aggregation", limit=25)

        assert isinstance(plan, PlanGraph)
        limit_step = next(step for step in plan.nodes if step.op == StepType.LIMIT)
        assert limit_step.params["n"] == 25

    def test_plan_from_macro_unknown_macro(self):
        """Test plan_from_macro with unknown macro."""
        planner = AgentPlanner()

        with pytest.raises(ValueError, match="Unknown macro"):
            planner.plan_from_macro("nonexistent_macro")


class TestNetworkResilience:
    """Test that system works without network/API keys."""

    def test_deterministic_planning_without_llm(self):
        """Test that deterministic planning works without LLM client."""
        planner = AgentPlanner(client=None)
        available_columns = ["pipeline_name", "eff_gas_day", "scheduled_quantity"]

        # This should work using fallback macros
        plan = planner.plan("find changepoints in the data", available_columns, fallback=True)

        assert isinstance(plan, PlanGraph)
        assert len(plan.nodes) > 0
        assert plan.inputs == ["raw"]

    def test_macro_creation_without_network(self):
        """Test that macro creation doesn't require network access."""
        # This should work completely offline
        plan = create_macro_plan("simple_aggregation")

        assert isinstance(plan, PlanGraph)
        assert len(plan.nodes) == 3

    def test_dsl_loader_without_network(self):
        """Test that DSL loader works without network access."""
        plan_data = {
            "nodes": [
                {
                    "id": "a",
                    "op": "aggregate",
                    "params": {
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                },
                {"id": "e", "op": "evidence_collect", "params": {}},
            ],
            "edges": [{"src": "raw", "dst": "a"}, {"src": "a", "dst": "e"}],
            "inputs": ["raw"],
            "outputs": ["a"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(plan_data, f)
            temp_path = f.name

        try:
            loader = DSLLoader()
            plan = loader.load_plan(temp_path)

            assert isinstance(plan, PlanGraph)
            # This validation should work completely offline
        finally:
            Path(temp_path).unlink()
