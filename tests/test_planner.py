"""Tests for the deterministic planner and LLM planner."""

import json
from unittest.mock import MagicMock, patch

from data_agent.core.plan_schema import Filter, Plan
from data_agent.core.planner import _validate_plan_json, plan


class TestDeterministicPlanner:
    """Test cases for deterministic template-based planning."""

    def test_sum_deliveries_on_date(self):
        """Test pattern: sum of deliveries for pipeline on date."""

        query = "sum of deliveries for ANR on 2022-01-01"
        result = plan(query, deterministic=True)

        assert isinstance(result, Plan)
        assert len(result.filters) == 3
        assert result.filters[0] == Filter(column="pipeline_name", op="=", value="ANR")
        assert result.filters[1] == Filter(column="rec_del_sign", op="=", value=-1)
        # Check that the date filter has the correct structure
        date_filter = result.filters[2]
        assert date_filter.column == "eff_gas_day"
        assert date_filter.op == "between"
        assert len(date_filter.value) == 2
        # Check that both values are Polars expressions
        from polars.expr.expr import Expr

        assert isinstance(date_filter.value[0], Expr)
        assert isinstance(date_filter.value[1], Expr)
        assert result.aggregate is not None
        assert result.aggregate.groupby == []
        assert result.aggregate.metrics == [{"col": "scheduled_quantity", "fn": "sum"}]

        # Validate JSON serialization
        json_data = result.model_dump()
        assert json_data["filters"][0]["column"] == "pipeline_name"
        assert json_data["aggregate"]["metrics"][0]["fn"] == "sum"

    def test_top_states_by_quantity_year(self):
        """Test pattern: top N states by total scheduled quantity in year."""

        query = "top 5 states by total scheduled quantity in 2022"
        result = plan(query, deterministic=True)

        assert isinstance(result, Plan)
        assert len(result.filters) == 1
        # Check that the date filter has the correct structure
        date_filter = result.filters[0]
        assert date_filter.column == "eff_gas_day"
        assert date_filter.op == "between"
        assert len(date_filter.value) == 2
        # Check that both values are Polars expressions
        from polars.expr.expr import Expr

        assert isinstance(date_filter.value[0], Expr)
        assert isinstance(date_filter.value[1], Expr)
        assert result.aggregate is not None
        assert result.aggregate.groupby == ["state_abb"]
        assert result.aggregate.metrics == [{"col": "scheduled_quantity", "fn": "sum"}]
        assert result.sort is not None
        assert result.sort.by == ["sum_scheduled_quantity"]
        assert result.sort.desc is True
        assert result.sort.limit == 5

        # Validate JSON serialization
        json_data = result.model_dump()
        assert json_data["sort"]["limit"] == 5
        assert json_data["aggregate"]["groupby"] == ["state_abb"]

    def test_deliveries_date_range(self):
        """Test pattern: deliveries for pipeline between dates."""

        query = "deliveries for TGP between 2022-01-01 and 2022-01-31"
        result = plan(query, deterministic=True)

        assert isinstance(result, Plan)
        assert len(result.filters) == 3
        assert result.filters[0] == Filter(column="pipeline_name", op="=", value="TGP")
        assert result.filters[1] == Filter(column="rec_del_sign", op="=", value=-1)
        # Check that the date filter has the correct structure
        date_filter = result.filters[2]
        assert date_filter.column == "eff_gas_day"
        assert date_filter.op == "between"
        assert len(date_filter.value) == 2
        # Check that both values are Polars expressions
        from polars.expr.expr import Expr

        assert isinstance(date_filter.value[0], Expr)
        assert isinstance(date_filter.value[1], Expr)
        assert result.aggregate is not None
        assert result.aggregate.groupby == ["eff_gas_day"]
        assert result.aggregate.metrics == [{"col": "scheduled_quantity", "fn": "sum"}]
        assert result.sort is not None
        assert result.sort.by == ["eff_gas_day"]
        assert result.sort.desc is False

        # Validate JSON serialization
        json_data = result.model_dump()
        # The JSON serialization will convert Polars expressions to strings
        # We can't easily test the exact string representation, so just check structure
        assert len(json_data["filters"][2]["value"]) == 2

    def test_total_receipts_pipeline(self):
        """Test pattern: total receipts for pipeline."""
        query = "total receipts for Kinder Morgan"
        result = plan(query, deterministic=True)

        assert isinstance(result, Plan)
        assert len(result.filters) == 2
        assert result.filters[0] == Filter(column="pipeline_name", op="=", value="Kinder Morgan")
        assert result.filters[1] == Filter(column="rec_del_sign", op="=", value=1)  # 1 for receipts
        assert result.aggregate is not None
        assert result.aggregate.groupby == []
        assert result.aggregate.metrics == [{"col": "scheduled_quantity", "fn": "sum"}]

        # Validate JSON serialization
        json_data = result.model_dump()
        assert json_data["filters"][1]["value"] == 1

    def test_average_scheduled_by_state(self):
        """Test pattern: average scheduled quantity by state."""
        query = "average scheduled quantity by state"
        result = plan(query, deterministic=True)

        assert isinstance(result, Plan)
        assert len(result.filters) == 0  # No filters for this pattern
        assert result.aggregate is not None
        assert result.aggregate.groupby == ["state_abb"]
        assert result.aggregate.metrics == [{"col": "scheduled_quantity", "fn": "avg"}]
        assert result.sort is not None
        assert result.sort.by == ["avg_scheduled_quantity"]
        assert result.sort.desc is True

        # Validate JSON serialization
        json_data = result.model_dump()
        assert json_data["aggregate"]["metrics"][0]["fn"] == "avg"

    def test_case_insensitive_matching(self):
        """Test that patterns work with different cases."""
        queries = [
            "SUM OF DELIVERIES FOR ANR ON 2022-01-01",
            "Sum Of Deliveries For ANR On 2022-01-01",
            "sum of deliveries for anr on 2022-01-01",
        ]

        for query in queries:
            result = plan(query, deterministic=True)
            assert isinstance(result, Plan)
            assert len(result.filters) == 3
            assert result.filters[0].value in [
                "ANR",
                "anr",
            ]  # Pipeline name preserves case from input

    def test_no_match_returns_empty_plan(self):
        """Test that unmatched queries return an empty plan."""
        query = "this query does not match any pattern"
        result = plan(query, deterministic=True)

        assert isinstance(result, Plan)
        assert len(result.filters) == 0
        assert result.aggregate is None
        assert result.sort is None
        assert result.op is None

    @patch("data_agent.core.planner._call_openai_planner")
    @patch("data_agent.core.planner._call_anthropic_planner")
    def test_llm_planning_fallback_when_no_api_keys(self, mock_anthropic, mock_openai):
        """Test that LLM-based planning falls back to deterministic when no API keys."""
        # Mock both APIs returning None (no API keys)
        mock_openai.return_value = None
        mock_anthropic.return_value = None

        # Use a query that matches deterministic pattern
        query = "sum of deliveries for ANR on 2022-01-01"
        result = plan(query, deterministic=False)

        # Should fall back to deterministic planning
        assert isinstance(result, Plan)
        assert len(result.filters) == 3
        assert result.filters[0].value == "ANR"

    def test_plan_schema_defaults(self):
        """Test that Plan schema has correct defaults."""
        empty_plan = Plan()

        assert empty_plan.filters == []
        assert empty_plan.resample is None
        assert empty_plan.aggregate is None
        assert empty_plan.sort is None
        assert empty_plan.op is None
        assert empty_plan.op_args == {}
        assert empty_plan.evidence is True
        assert empty_plan.format == "table"

    def test_complex_pipeline_names(self):
        """Test that complex pipeline names are handled correctly."""
        query = "sum of deliveries for Texas Eastern Transmission Pipeline on 2022-01-01"
        result = plan(query, deterministic=True)

        assert isinstance(result, Plan)
        assert result.filters[0].value == "Texas Eastern Transmission Pipeline"

        # Test with abbreviations
        query2 = "sum of deliveries for TGP-100 on 2022-01-01"
        result2 = plan(query2, deterministic=True)
        assert result2.filters[0].value == "TGP-100"


class TestLLMPlanner:
    """Test cases for LLM-based planning."""

    def test_validate_plan_json_valid(self):
        """Test that valid plan JSON passes validation."""
        valid_plan = {
            "filters": [
                {"column": "pipeline_name", "op": "=", "value": "ANR"},
                {"column": "rec_del_sign", "op": "=", "value": -1},
            ],
            "aggregate": {
                "groupby": ["state_abb"],
                "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
            },
            "sort": {"by": ["sum_scheduled_quantity"], "desc": True, "limit": 10},
        }

        assert _validate_plan_json(valid_plan) is True

    def test_validate_plan_json_invalid_column(self):
        """Test that invalid column names fail validation."""
        invalid_plan = {"filters": [{"column": "invalid_column", "op": "=", "value": "test"}]}

        assert _validate_plan_json(invalid_plan) is False

    def test_validate_plan_json_invalid_op(self):
        """Test that invalid operations fail validation."""
        invalid_plan = {
            "filters": [{"column": "pipeline_name", "op": "invalid_op", "value": "test"}]
        }

        assert _validate_plan_json(invalid_plan) is False

    def test_validate_plan_json_invalid_metric_function(self):
        """Test that invalid metric functions fail validation."""
        invalid_plan = {
            "aggregate": {
                "groupby": ["state_abb"],
                "metrics": [{"col": "scheduled_quantity", "fn": "invalid_fn"}],
            }
        }

        assert _validate_plan_json(invalid_plan) is False

    def test_validate_plan_json_allows_computed_columns(self):
        """Test that computed column names in sort are allowed."""
        valid_plan = {
            "aggregate": {
                "groupby": ["state_abb"],
                "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
            },
            "sort": {"by": ["sum_scheduled_quantity"], "desc": True},
        }

        assert _validate_plan_json(valid_plan) is True

    @patch("data_agent.core.planner._call_openai_planner")
    @patch("data_agent.core.planner._call_anthropic_planner")
    def test_llm_planner_openai_success(self, mock_anthropic, mock_openai):
        """Test LLM planner with successful OpenAI response."""
        # Mock successful OpenAI response
        mock_plan_data = {
            "filters": [
                {"column": "pipeline_name", "op": "=", "value": "ANR"},
                {"column": "rec_del_sign", "op": "=", "value": -1},
            ],
            "aggregate": {"groupby": [], "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]},
        }
        mock_openai.return_value = mock_plan_data
        mock_anthropic.return_value = None  # Should not be called

        result = plan("sum deliveries for ANR pipeline", deterministic=False)

        assert isinstance(result, Plan)
        assert len(result.filters) == 2
        assert result.filters[0].column == "pipeline_name"
        assert result.filters[0].value == "ANR"
        assert result.aggregate is not None
        assert result.aggregate.metrics[0]["fn"] == "sum"

        # Verify OpenAI was called but not Anthropic
        mock_openai.assert_called_once()
        mock_anthropic.assert_not_called()

    @patch("data_agent.core.planner._call_openai_planner")
    @patch("data_agent.core.planner._call_anthropic_planner")
    def test_llm_planner_anthropic_fallback(self, mock_anthropic, mock_openai):
        """Test LLM planner falls back to Anthropic when OpenAI fails."""
        # Mock OpenAI failure and Anthropic success
        mock_openai.return_value = None
        mock_plan_data = {
            "filters": [{"column": "state_abb", "op": "=", "value": "TX"}],
            "aggregate": {
                "groupby": ["pipeline_name"],
                "metrics": [{"col": "scheduled_quantity", "fn": "avg"}],
            },
        }
        mock_anthropic.return_value = mock_plan_data

        result = plan("average flow by pipeline in Texas", deterministic=False)

        assert isinstance(result, Plan)
        assert len(result.filters) == 1
        assert result.filters[0].column == "state_abb"
        assert result.filters[0].value == "TX"
        assert result.aggregate is not None
        assert result.aggregate.metrics[0]["fn"] == "avg"

        # Verify both were called
        mock_openai.assert_called_once()
        mock_anthropic.assert_called_once()

    @patch("data_agent.core.planner._call_openai_planner")
    @patch("data_agent.core.planner._call_anthropic_planner")
    def test_llm_planner_fallback_to_deterministic(self, mock_anthropic, mock_openai):
        """Test LLM planner falls back to deterministic when both LLMs fail."""
        # Mock both LLMs failing
        mock_openai.return_value = None
        mock_anthropic.return_value = None

        # Use a query that matches a deterministic pattern
        result = plan("sum of deliveries for ANR on 2022-01-01", deterministic=False)

        # Should fall back to deterministic planning
        assert isinstance(result, Plan)
        assert len(result.filters) == 3
        assert result.filters[0].value == "ANR"
        assert result.filters[1].value == -1  # deliveries

        # Verify both LLMs were attempted
        mock_openai.assert_called_once()
        mock_anthropic.assert_called_once()

    @patch("data_agent.core.planner._call_openai_planner")
    @patch("data_agent.core.planner._call_anthropic_planner")
    def test_llm_planner_invalid_response_fallback(self, mock_anthropic, mock_openai):
        """Test LLM planner falls back when response is invalid."""
        # Mock invalid response (invalid column)
        invalid_plan_data = {"filters": [{"column": "invalid_column", "op": "=", "value": "test"}]}
        mock_openai.return_value = invalid_plan_data
        mock_anthropic.return_value = None

        # Use a query that matches a deterministic pattern for fallback
        result = plan("sum of deliveries for ANR on 2022-01-01", deterministic=False)

        # Should fall back to deterministic planning
        assert isinstance(result, Plan)
        assert len(result.filters) == 3
        assert result.filters[0].value == "ANR"

        # Verify both LLMs were attempted
        mock_openai.assert_called_once()
        mock_anthropic.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_call_openai_planner_integration(self, mock_openai_class):
        """Test OpenAI API call integration."""
        from data_agent.core.planner import _call_openai_planner

        # Mock the OpenAI client and response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]

        # Mock tool_calls structure
        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = json.dumps(
            {
                "filters": [{"column": "pipeline_name", "op": "=", "value": "ANR"}],
                "aggregate": {
                    "groupby": [],
                    "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                },
            }
        )
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_client.chat.completions.create.return_value = mock_response

        result = _call_openai_planner("test query")

        assert result is not None
        assert "filters" in result
        assert len(result["filters"]) == 1
        assert result["filters"][0]["column"] == "pipeline_name"

        # Verify the client was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4"
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["tool_choice"]["function"]["name"] == "generate_query_plan"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("anthropic.Anthropic")
    def test_call_anthropic_planner_integration(self, mock_anthropic_class):
        """Test Anthropic API call integration."""
        from data_agent.core.planner import _call_anthropic_planner

        # Mock the Anthropic client and response
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_content = MagicMock()
        mock_content.type = "tool_use"
        mock_content.input = {
            "filters": [{"column": "state_abb", "op": "=", "value": "TX"}],
            "aggregate": {
                "groupby": ["pipeline_name"],
                "metrics": [{"col": "scheduled_quantity", "fn": "avg"}],
            },
        }

        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message

        result = _call_anthropic_planner("test query")

        assert result is not None
        assert "filters" in result
        assert len(result["filters"]) == 1
        assert result["filters"][0]["column"] == "state_abb"

        # Verify the client was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-sonnet-20240229"
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["tool_choice"]["name"] == "generate_query_plan"
