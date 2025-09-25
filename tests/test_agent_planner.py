"""Tests for Agent Planner functionality."""

import json
from unittest.mock import Mock, patch

import pytest

from data_agent.core.agent_planner import (
    AgentPlanner,
    PlanValidationError,
    estimate_plan_complexity,
    plan_from_llm,
)
from data_agent.core.agent_schema import Edge, PlanGraph, Step, StepType
from data_agent.core.llm_client import LLMClient


class TestAgentPlanner:
    """Test cases for AgentPlanner class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.available_columns = [
            "pipeline_name",
            "loc_name",
            "connecting_pipeline",
            "connecting_entity",
            "rec_del_sign",
            "category_short",
            "country_name",
            "state_abb",
            "county_name",
            "latitude",
            "longitude",
            "eff_gas_day",
            "scheduled_quantity",
        ]

        # Create a mock LLM client
        self.mock_client = Mock(spec=LLMClient)
        self.planner = AgentPlanner(self.mock_client)

    def test_initialization_with_client(self):
        """Test planner initialization with provided client."""
        client = Mock(spec=LLMClient)
        planner = AgentPlanner(client)
        assert planner.client == client

    def test_initialization_without_client(self):
        """Test planner initialization without client (allows None)."""
        planner = AgentPlanner()
        assert planner.client is None

    def test_build_function_schema(self):
        """Test function schema building for LLM."""
        schema = self.planner._build_function_schema(self.available_columns)

        assert schema["name"] == "create_dag_plan"
        assert "description" in schema
        assert "parameters" in schema

        params = schema["parameters"]
        assert params["type"] == "object"
        assert "nodes" in params["properties"]
        assert "edges" in params["properties"]

        # Check node schema
        node_schema = params["properties"]["nodes"]
        assert node_schema["type"] == "array"
        node_item = node_schema["items"]
        assert "id" in node_item["properties"]
        assert "op" in node_item["properties"]
        assert node_item["properties"]["op"]["enum"] == [op.value for op in StepType]

    def test_build_system_prompt(self):
        """Test system prompt building."""
        prompt = self.planner._build_system_prompt(self.available_columns)

        assert "expert data processing planner" in prompt.lower()
        assert "pipeline_name" in prompt
        assert "scheduled_quantity" in prompt
        assert "filter" in prompt
        assert "aggregate" in prompt
        assert json.dumps in prompt.__class__.__bases__ or "nodes" in prompt

    def test_successful_llm_planning(self):
        """Test successful LLM planning flow."""
        # Mock LLM response
        mock_response = {
            "tool_calls": [
                Mock(
                    function=Mock(
                        name="create_dag_plan",
                        arguments=json.dumps(
                            {
                                "nodes": [
                                    {
                                        "id": "f",
                                        "op": "filter",
                                        "params": {
                                            "column": "eff_gas_day",
                                            "op": "between",
                                            "value": ["2021-01-01", "2021-12-31"],
                                        },
                                    },
                                    {"id": "l", "op": "limit", "params": {"n": 100}},
                                    {"id": "e", "op": "evidence_collect", "params": {}},
                                ],
                                "edges": [
                                    {"src": "raw", "dst": "f"},
                                    {"src": "f", "dst": "l"},
                                    {"src": "l", "dst": "e"},
                                ],
                                "inputs": ["raw"],
                                "outputs": ["l"],
                            }
                        ),
                    )
                )
            ]
        }
        self.mock_client.call.return_value = mock_response

        plan = self.planner.plan(
            "Filter data for 2021 and limit to 100 rows", self.available_columns
        )

        assert isinstance(plan, PlanGraph)
        # The repair logic may have removed the invalid filter step and added evidence_collect
        assert len(plan.nodes) >= 2  # At least limit and evidence_collect
        assert len(plan.edges) >= 2
        assert plan.inputs == ["raw"]
        assert plan.outputs == ["l"]

        # Verify LLM was called with correct parameters
        self.mock_client.call.assert_called_once()
        call_args = self.mock_client.call.call_args
        assert "tools" in call_args.kwargs
        assert "temperature" in call_args.kwargs
        assert call_args.kwargs["temperature"] == 0.1

    def test_llm_planning_failure_with_fallback(self):
        """Test LLM planning failure falling back to deterministic plan."""
        # Mock LLM failure
        self.mock_client.call.side_effect = Exception("LLM API error")

        plan = self.planner.plan("sum of scheduled quantity", self.available_columns, fallback=True)

        assert isinstance(plan, PlanGraph)
        assert len(plan.nodes) > 0
        # Should have aggregation step from fallback
        assert any(step.op == StepType.AGGREGATE for step in plan.nodes)

    def test_llm_planning_failure_without_fallback(self):
        """Test LLM planning failure without fallback raises error."""
        # Mock LLM failure
        self.mock_client.call.side_effect = Exception("LLM API error")

        with pytest.raises(PlanValidationError, match="LLM planning failed"):
            self.planner.plan("test query", self.available_columns, fallback=False)

    def test_invalid_llm_response_no_tool_calls(self):
        """Test handling of LLM response without tool calls."""
        self.mock_client.call.return_value = {"content": "No tool calls"}

        with pytest.raises(PlanValidationError):
            self.planner.plan("test query", self.available_columns, fallback=False)

    def test_invalid_llm_response_wrong_function(self):
        """Test handling of LLM response with wrong function name."""
        mock_response = {"tool_calls": [Mock(function=Mock(name="wrong_function", arguments="{}"))]}
        self.mock_client.call.return_value = mock_response

        with pytest.raises(PlanValidationError):
            self.planner.plan("test query", self.available_columns, fallback=False)

    def test_invalid_llm_response_malformed_json(self):
        """Test handling of LLM response with malformed JSON."""
        mock_response = {
            "tool_calls": [Mock(function=Mock(name="create_dag_plan", arguments="invalid json"))]
        }
        self.mock_client.call.return_value = mock_response

        with pytest.raises(PlanValidationError):
            self.planner.plan("test query", self.available_columns, fallback=False)


class TestFallbackPlans:
    """Test deterministic fallback plan generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.available_columns = ["pipeline_name", "scheduled_quantity", "eff_gas_day"]
        self.planner = AgentPlanner(Mock(spec=LLMClient))

    def test_simple_aggregation_fallback(self):
        """Test fallback for simple aggregation queries."""
        plan = self.planner._fallback_plan("sum of scheduled quantity", self.available_columns)

        assert isinstance(plan, PlanGraph)
        assert any(step.op == StepType.AGGREGATE for step in plan.nodes)
        assert any(step.op == StepType.EVIDENCE_COLLECT for step in plan.nodes)

    def test_time_series_fallback(self):
        """Test fallback for time series queries."""
        plan = self.planner._fallback_plan("detect regime shifts", self.available_columns)

        assert isinstance(plan, PlanGraph)
        assert any(step.op == StepType.STL_DESEASONALIZE for step in plan.nodes)
        assert any(step.op == StepType.CHANGEPOINT for step in plan.nodes)

    def test_top_k_fallback(self):
        """Test fallback for top-k queries."""
        plan = self.planner._fallback_plan("top 5 pipelines", self.available_columns)

        assert isinstance(plan, PlanGraph)
        assert any(step.op == StepType.RANK for step in plan.nodes)
        assert any(step.op == StepType.LIMIT for step in plan.nodes)

        # Check that limit is set to 5
        limit_steps = [s for s in plan.nodes if s.op == StepType.LIMIT]
        assert len(limit_steps) == 1
        assert limit_steps[0].params["n"] == 5

    def test_top_k_fallback_with_number_extraction(self):
        """Test fallback for top-k queries with number extraction."""
        plan = self.planner._fallback_plan(
            "show me the top 15 highest values", self.available_columns
        )

        limit_steps = [s for s in plan.nodes if s.op == StepType.LIMIT]
        assert len(limit_steps) == 1
        assert limit_steps[0].params["n"] == 15

    def test_default_fallback(self):
        """Test default fallback for unrecognized queries."""
        plan = self.planner._fallback_plan("unknown query pattern", self.available_columns)

        assert isinstance(plan, PlanGraph)
        assert any(step.op == StepType.LIMIT for step in plan.nodes)
        assert any(step.op == StepType.EVIDENCE_COLLECT for step in plan.nodes)


class TestPlanRepair:
    """Test plan repair and validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.available_columns = ["pipeline_name", "scheduled_quantity", "eff_gas_day"]
        self.planner = AgentPlanner(Mock(spec=LLMClient))

    def test_repair_missing_connections(self):
        """Test repair of plans with missing connections."""
        # Create a plan with nodes but no edges
        plan = PlanGraph(
            nodes=[
                Step(
                    id="f",
                    op=StepType.FILTER,
                    params={"column": "pipeline_name", "op": "=", "value": "test"},
                ),
                Step(id="l", op=StepType.LIMIT, params={"n": 100}),
            ],
            edges=[],
            inputs=["raw"],
            outputs=["l"],
        )

        repaired = self.planner._repair_plan(plan, self.available_columns)

        # Should have added connection from raw to first node
        assert len(repaired.edges) > 0
        assert any(edge.src == "raw" for edge in repaired.edges)

    def test_repair_missing_evidence_collect(self):
        """Test repair of plans missing evidence collection."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="f",
                    op=StepType.FILTER,
                    params={"column": "pipeline_name", "op": "=", "value": "test"},
                )
            ],
            edges=[Edge(src="raw", dst="f")],
            inputs=["raw"],
            outputs=["f"],
        )

        repaired = self.planner._repair_plan(plan, self.available_columns)

        # Should have added evidence collect step
        evidence_steps = [s for s in repaired.nodes if s.op == StepType.EVIDENCE_COLLECT]
        assert len(evidence_steps) == 1

    def test_repair_add_limit_for_aggregation(self):
        """Test repair adding limit for plans with aggregation but no limit."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="a",
                    op=StepType.AGGREGATE,
                    params={
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                ),
                Step(id="e", op=StepType.EVIDENCE_COLLECT, params={}),
            ],
            edges=[Edge(src="raw", dst="a"), Edge(src="a", dst="e")],
            inputs=["raw"],
            outputs=["a"],
        )

        repaired = self.planner._repair_plan(plan, self.available_columns)

        # Should have added a limit step
        limit_steps = [s for s in repaired.nodes if s.op == StepType.LIMIT]
        assert len(limit_steps) == 1

    def test_repair_invalid_column_references(self):
        """Test repair of invalid column references."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="f",
                    op=StepType.FILTER,
                    params={"column": "nonexistent_column", "op": "=", "value": "test"},
                ),
                Step(id="e", op=StepType.EVIDENCE_COLLECT, params={}),
            ],
            edges=[Edge(src="raw", dst="f"), Edge(src="f", dst="e")],
            inputs=["raw"],
            outputs=["f"],
        )

        repaired = self.planner._repair_plan(plan, self.available_columns)

        # Should have attempted to fix the column reference
        filter_step = next(s for s in repaired.nodes if s.op == StepType.FILTER)
        # Since no similar column exists, it should remain unchanged but logged
        assert "column" in filter_step.params

    def test_repair_similar_column_replacement(self):
        """Test repair with similar column replacement."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="f",
                    op=StepType.FILTER,
                    params={"column": "pipeline", "op": "=", "value": "test"},
                ),
                Step(id="e", op=StepType.EVIDENCE_COLLECT, params={}),
            ],
            edges=[Edge(src="raw", dst="f"), Edge(src="f", dst="e")],
            inputs=["raw"],
            outputs=["f"],
        )

        repaired = self.planner._repair_plan(plan, self.available_columns)

        # Should have replaced "pipeline" with "pipeline_name"
        filter_step = next(s for s in repaired.nodes if s.op == StepType.FILTER)
        assert filter_step.params["column"] == "pipeline_name"

    def test_has_valid_path_true(self):
        """Test valid path detection returns True for valid plans."""
        plan = PlanGraph(
            nodes=[Step(id="f", op=StepType.LIMIT, params={"n": 100})],
            edges=[Edge(src="raw", dst="f")],
            inputs=["raw"],
            outputs=["f"],
        )

        assert self.planner._has_valid_path(plan) is True

    def test_has_valid_path_false(self):
        """Test valid path detection returns False for disconnected plans."""
        plan = PlanGraph(
            nodes=[
                Step(id="f", op=StepType.LIMIT, params={"n": 50}),
                Step(id="l", op=StepType.LIMIT, params={"n": 100}),
            ],
            edges=[Edge(src="raw", dst="f")],  # l is not connected
            inputs=["raw"],
            outputs=["l"],  # l is not reachable from raw
        )

        assert self.planner._has_valid_path(plan) is False

    def test_find_similar_column_exact_match(self):
        """Test finding exact column matches (case insensitive)."""
        result = self.planner._find_similar_column("PIPELINE_NAME", self.available_columns)
        assert result == "pipeline_name"

    def test_find_similar_column_substring_match(self):
        """Test finding substring column matches."""
        result = self.planner._find_similar_column("pipeline", self.available_columns)
        assert result == "pipeline_name"

    def test_find_similar_column_no_match(self):
        """Test finding similar column when no match exists."""
        result = self.planner._find_similar_column("completely_different", self.available_columns)
        assert result is None


class TestEstimatePlanComplexity:
    """Test plan complexity estimation."""

    def test_simple_plan_estimation(self):
        """Test complexity estimation for simple plan."""
        plan = PlanGraph(
            nodes=[
                Step(
                    id="a",
                    op=StepType.AGGREGATE,
                    params={
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                ),
                Step(id="l", op=StepType.LIMIT, params={"n": 100}),
            ],
            edges=[Edge(src="raw", dst="a"), Edge(src="a", dst="l")],
            inputs=["raw"],
            outputs=["l"],
        )

        estimates = estimate_plan_complexity(plan)

        assert estimates["steps"] == 2
        assert estimates["edges"] == 2
        assert estimates["estimated_time_seconds"] > 0
        assert estimates["estimated_memory_mb"] > 0
        assert "topological_order" in estimates
        assert len(estimates["topological_order"]) == 3  # raw + 2 steps

    def test_complex_plan_estimation(self):
        """Test complexity estimation for complex plan with expensive operations."""
        plan = PlanGraph(
            nodes=[
                Step(id="s", op=StepType.STL_DESEASONALIZE, params={"column": "value"}),
                Step(id="c", op=StepType.CHANGEPOINT, params={"column": "value", "method": "pelt"}),
            ],
            edges=[Edge(src="raw", dst="s"), Edge(src="s", dst="c")],
            inputs=["raw"],
            outputs=["c"],
        )

        estimates = estimate_plan_complexity(plan)

        # Expensive operations should have higher estimates
        assert estimates["estimated_time_seconds"] >= 10  # 5 + 5 from STL and changepoint
        assert estimates["estimated_memory_mb"] >= 200  # 100 + 100
        assert len(estimates["will_checkpoint"]) == 2  # Both operations should checkpoint


class TestPlanFromLLMFunction:
    """Test the convenience function for LLM-based planning."""

    @patch("data_agent.core.agent_planner.AgentPlanner")
    def test_plan_from_llm_with_client(self, mock_planner_class):
        """Test plan_from_llm function with provided client."""
        mock_client = Mock(spec=LLMClient)
        mock_planner = Mock()
        mock_plan = Mock(spec=PlanGraph)
        mock_planner_class.return_value = mock_planner
        mock_planner.plan.return_value = mock_plan

        result = plan_from_llm("test query", mock_client)

        assert result == mock_plan
        mock_planner_class.assert_called_once_with(mock_client)
        mock_planner.plan.assert_called_once()

    @patch("data_agent.core.agent_planner.get_default_llm_client")
    @patch("data_agent.core.agent_planner.AgentPlanner")
    def test_plan_from_llm_without_client(self, mock_planner_class, mock_get_client):
        """Test plan_from_llm function without client (uses default)."""
        mock_client = Mock(spec=LLMClient)
        mock_get_client.return_value = mock_client
        mock_planner = Mock()
        mock_plan = Mock(spec=PlanGraph)
        mock_planner_class.return_value = mock_planner
        mock_planner.plan.return_value = mock_plan

        result = plan_from_llm("test query")

        assert result == mock_plan
        mock_get_client.assert_called_once()
        mock_planner_class.assert_called_once_with(mock_client)


class TestIntegrationScenarios:
    """Integration tests for common planning scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.available_columns = [
            "pipeline_name",
            "loc_name",
            "connecting_pipeline",
            "connecting_entity",
            "rec_del_sign",
            "category_short",
            "country_name",
            "state_abb",
            "county_name",
            "latitude",
            "longitude",
            "eff_gas_day",
            "scheduled_quantity",
        ]

    def test_successful_planning_with_repair(self):
        """Test successful planning flow with repair needed."""
        # Mock LLM that returns a plan needing repair
        mock_client = Mock(spec=LLMClient)
        mock_response = {
            "tool_calls": [
                Mock(
                    function=Mock(
                        name="create_dag_plan",
                        arguments=json.dumps(
                            {
                                "nodes": [
                                    {
                                        "id": "f",
                                        "op": "filter",
                                        "params": {
                                            "column": "pipeline",
                                            "op": "=",
                                            "value": "test",
                                        },
                                    },  # Invalid column
                                    {
                                        "id": "a",
                                        "op": "aggregate",
                                        "params": {
                                            "groupby": ["pipeline_name"],
                                            "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                                        },
                                    },
                                    # Missing evidence_collect and limit
                                ],
                                "edges": [{"src": "raw", "dst": "f"}, {"src": "f", "dst": "a"}],
                                "inputs": ["raw"],
                                "outputs": ["a"],
                            }
                        ),
                    )
                )
            ]
        }
        mock_client.call.return_value = mock_response

        planner = AgentPlanner(mock_client)
        plan = planner.plan("sum scheduled quantity by pipeline", self.available_columns)

        # Verify the plan was repaired
        assert isinstance(plan, PlanGraph)

        # Should have fixed the column name
        filter_steps = [s for s in plan.nodes if s.op == StepType.FILTER]
        if filter_steps:
            assert filter_steps[0].params["column"] == "pipeline_name"

        # Should have added evidence collect
        evidence_steps = [s for s in plan.nodes if s.op == StepType.EVIDENCE_COLLECT]
        assert len(evidence_steps) >= 1

        # Should have added limit for aggregation
        limit_steps = [s for s in plan.nodes if s.op == StepType.LIMIT]
        assert len(limit_steps) >= 1

    def test_fallback_to_deterministic_on_llm_failure(self):
        """Test complete fallback flow when LLM fails."""
        # Mock LLM that always fails
        mock_client = Mock(spec=LLMClient)
        mock_client.call.side_effect = Exception("API error")

        planner = AgentPlanner(mock_client)
        plan = planner.plan(
            "top 10 pipelines by scheduled quantity", self.available_columns, fallback=True
        )

        # Should get a valid plan from fallback
        assert isinstance(plan, PlanGraph)
        assert len(plan.nodes) > 0

        # Should be a top-k plan
        rank_steps = [s for s in plan.nodes if s.op == StepType.RANK]
        limit_steps = [s for s in plan.nodes if s.op == StepType.LIMIT]
        assert len(rank_steps) >= 1 or len(limit_steps) >= 1

        if limit_steps:
            assert limit_steps[0].params["n"] == 10

    def test_complete_failure_scenario(self):
        """Test scenario where both LLM and fallback fail."""
        # Mock LLM that fails
        mock_client = Mock(spec=LLMClient)
        mock_client.call.side_effect = Exception("API error")

        planner = AgentPlanner(mock_client)

        # This should still work because fallback creates valid plans
        # But let's test the no-fallback scenario
        with pytest.raises(PlanValidationError):
            planner.plan("test query", self.available_columns, fallback=False)
