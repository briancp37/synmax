"""Tests for the deterministic planner."""

import pytest

from data_agent.core.plan_schema import Filter, Plan
from data_agent.core.planner import plan


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

    def test_llm_planning_not_implemented(self):
        """Test that LLM-based planning raises NotImplementedError."""
        query = "any query"
        with pytest.raises(NotImplementedError, match="LLM-based planning not yet implemented"):
            plan(query, deterministic=False)

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
