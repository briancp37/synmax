"""Tests for LLM Answer Composer."""

from unittest.mock import Mock

import polars as pl

from data_agent.core.answer import compose_answer_llm


class TestComposeAnswerLLM:
    """Test suite for compose_answer_llm function."""

    def test_empty_dataframe_returns_no_results_message(self):
        """Test that empty DataFrame returns the expected message."""
        plan = {"op": "top_k"}
        df = pl.DataFrame()  # Empty DataFrame
        evidence = {"filters": []}
        llm = Mock()

        result = compose_answer_llm(plan, df, evidence, llm)

        assert result == "No results for the specified filters; see Evidence."
        llm.call.assert_not_called()

    def test_top_k_query_includes_entity_and_value(self):
        """Test that top-k query answer includes entity name and numeric value."""
        plan = {"op": "top_k"}
        df = pl.DataFrame(
            {"state_abb": ["TX", "CA", "FL"], "total_quantity": [1000.0, 800.0, 600.0]}
        )
        evidence = {"filters": [{"column": "year", "op": "=", "value": 2022}]}

        # Mock LLM response
        llm = Mock()
        llm.call.return_value = {
            "content": "Texas leads with 1000.0 total quantity, followed by California at 800.0."
        }

        result = compose_answer_llm(plan, df, evidence, llm)

        # Verify LLM was called
        assert llm.call.called
        call_args = llm.call.call_args[1]["messages"][0]["content"]

        # Check that the top entity and value are in the prompt
        assert "TX" in call_args
        assert "1000.0" in call_args

        # Check the returned answer contains expected content
        assert "Texas" in result
        assert "1000.0" in result

    def test_stl_changepoint_includes_entity_date_magnitude(self):
        """Test that STL+changepoint answer includes entity, date, and magnitude."""
        plan = {"op": "changepoint", "method": "STL"}
        df = pl.DataFrame(
            {
                "pipeline_name": ["ANR Pipeline"],
                "changepoint_date": ["2022-03-15"],
                "magnitude": [250.5],
            }
        )
        evidence = {"filters": [{"column": "pipeline_name", "op": "=", "value": "ANR Pipeline"}]}

        llm = Mock()
        llm.call.return_value = {
            "content": (
                "ANR Pipeline experienced a significant changepoint on 2022-03-15 "
                "with magnitude 250.5 using STL analysis."
            )
        }

        result = compose_answer_llm(plan, df, evidence, llm)

        # Verify the prompt contains key data
        call_args = llm.call.call_args[1]["messages"][0]["content"]
        assert "ANR Pipeline" in call_args
        assert "2022-03-15" in call_args
        assert "250.5" in call_args

        # Verify answer contains expected elements
        assert "ANR Pipeline" in result
        assert "2022-03-15" in result
        assert "250.5" in result

    def test_prompt_structure_contains_required_elements(self):
        """Test that the prompt contains all required elements."""
        plan = {"op": "aggregate", "macro": "sum_by_state"}
        df = pl.DataFrame({"state": ["TX", "CA"], "value": [100, 200]})
        evidence = {
            "filters": [{"column": "year", "op": "between", "value": ["2022-01-01", "2022-12-31"]}]
        }

        llm = Mock()
        llm.call.return_value = {"content": "Test answer"}

        compose_answer_llm(plan, df, evidence, llm, style="detailed")

        # Check the prompt structure
        call_args = llm.call.call_args[1]["messages"][0]["content"]

        # Verify required elements are present
        assert "Task: aggregate" in call_args
        assert "Filters applied:" in call_args
        assert "Method used: aggregate" in call_args
        assert "Result columns:" in call_args
        assert "Top results:" in call_args
        assert "Instructions:" in call_args
        assert "Style: detailed" in call_args

        # Verify JSON structure for highlights
        assert '"state": "TX"' in call_args
        assert '"value": 100' in call_args

    def test_limits_to_top_3_rows(self):
        """Test that only top 3 rows are included in highlights."""
        plan = {"op": "top_k"}
        df = pl.DataFrame({"entity": ["A", "B", "C", "D", "E"], "value": [100, 90, 80, 70, 60]})
        evidence = {"filters": []}

        llm = Mock()
        llm.call.return_value = {"content": "Test answer"}

        compose_answer_llm(plan, df, evidence, llm)

        call_args = llm.call.call_args[1]["messages"][0]["content"]

        # Should include first 3 entities
        assert '"entity": "A"' in call_args
        assert '"entity": "B"' in call_args
        assert '"entity": "C"' in call_args

        # Should not include 4th and 5th entities
        assert '"entity": "D"' not in call_args
        assert '"entity": "E"' not in call_args

    def test_handles_llm_exception_gracefully(self):
        """Test that LLM exceptions are handled gracefully."""
        plan = {"op": "test"}
        df = pl.DataFrame({"col": [1, 2, 3]})
        evidence = {"filters": []}

        llm = Mock()
        llm.call.side_effect = Exception("API Error")

        result = compose_answer_llm(plan, df, evidence, llm)

        assert result == "Answer unavailable."

    def test_handles_missing_content_in_response(self):
        """Test handling of LLM response without content key."""
        plan = {"op": "test"}
        df = pl.DataFrame({"col": [1, 2, 3]})
        evidence = {"filters": []}

        llm = Mock()
        llm.call.return_value = {}  # No 'content' key

        result = compose_answer_llm(plan, df, evidence, llm)

        assert result == "Answer unavailable."

    def test_uses_fallback_task_names(self):
        """Test that function uses fallback task names when op is missing."""
        # Test with macro
        plan = {"macro": "daily_totals"}
        df = pl.DataFrame({"value": [100]})
        evidence = {"filters": []}

        llm = Mock()
        llm.call.return_value = {"content": "Test"}

        compose_answer_llm(plan, df, evidence, llm)

        call_args = llm.call.call_args[1]["messages"][0]["content"]
        assert "Task: daily_totals" in call_args

        # Test with neither op nor macro
        plan = {}
        compose_answer_llm(plan, df, evidence, llm)

        call_args = llm.call.call_args[1]["messages"][0]["content"]
        assert "Task: analysis" in call_args
