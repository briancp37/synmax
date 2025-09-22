"""Tests for executor and evidence functionality."""

import polars as pl

from data_agent.core.executor import Answer, run
from data_agent.core.plan_schema import Aggregate, Filter, Plan, Sort


def test_executor_basic_filter():
    """Test basic filtering functionality."""
    # Create test data
    data = {
        "pipeline_name": ["ANR", "ANR", "TGP", "TGP"],
        "state_abb": ["TX", "LA", "TX", "LA"],
        "scheduled_quantity": [100.0, 200.0, 150.0, 250.0],
        "eff_gas_day": ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"],
    }
    df = pl.DataFrame(data)
    lf = df.lazy()

    # Create plan with filter
    plan = Plan(
        filters=[Filter(column="pipeline_name", op="=", value="ANR")],
        aggregate=Aggregate(
            groupby=["state_abb"], metrics=[{"col": "scheduled_quantity", "fn": "sum"}]
        ),
    )

    # Execute
    result = run(lf, plan)

    # Verify results
    assert isinstance(result, Answer)
    assert result.table.height == 2  # TX and LA for ANR
    assert "state_abb" in result.table.columns
    assert "sum_scheduled_quantity" in result.table.columns

    # Check evidence
    assert "filters" in result.evidence
    assert "aggregate" in result.evidence
    assert "rows_out" in result.evidence
    assert result.evidence["rows_out"] == 2


def test_executor_between_filter():
    """Test between filter functionality."""
    data = {
        "scheduled_quantity": [50.0, 100.0, 150.0, 200.0, 250.0],
        "eff_gas_day": ["2022-01-01"] * 5,
    }
    df = pl.DataFrame(data)
    lf = df.lazy()

    plan = Plan(
        filters=[Filter(column="scheduled_quantity", op="between", value=[100.0, 200.0])],
        aggregate=Aggregate(metrics=[{"col": "scheduled_quantity", "fn": "count"}]),
    )

    result = run(lf, plan)

    # Should have 3 rows (100, 150, 200)
    assert result.table.height == 1
    assert result.table["count"][0] == 3


def test_executor_in_filter():
    """Test in filter functionality."""
    data = {
        "pipeline_name": ["ANR", "TGP", "KMI", "ANR"],
        "scheduled_quantity": [100.0, 200.0, 300.0, 400.0],
    }
    df = pl.DataFrame(data)
    lf = df.lazy()

    plan = Plan(
        filters=[Filter(column="pipeline_name", op="in", value=["ANR", "TGP"])],
        aggregate=Aggregate(metrics=[{"col": "scheduled_quantity", "fn": "sum"}]),
    )

    result = run(lf, plan)

    # Should sum ANR (100+400) and TGP (200) = 700
    assert result.table.height == 1
    assert result.table["sum_scheduled_quantity"][0] == 700.0


def test_executor_sort_and_limit():
    """Test sorting and limiting functionality."""
    data = {
        "state_abb": ["TX", "LA", "CA", "NY", "FL"],
        "scheduled_quantity": [500.0, 400.0, 300.0, 200.0, 100.0],
    }
    df = pl.DataFrame(data)
    lf = df.lazy()

    plan = Plan(
        aggregate=Aggregate(
            groupby=["state_abb"], metrics=[{"col": "scheduled_quantity", "fn": "sum"}]
        ),
        sort=Sort(by=["sum_scheduled_quantity"], desc=True, limit=3),
    )

    result = run(lf, plan)

    # Should have top 3 states by quantity
    assert result.table.height == 3
    assert result.table["state_abb"][0] == "TX"  # Highest
    assert result.table["state_abb"][1] == "LA"  # Second highest
    assert result.table["state_abb"][2] == "CA"  # Third highest


def test_executor_golden_dataset_integration():
    """Integration test using golden dataset."""
    # Load golden dataset
    lf = pl.scan_parquet("examples/golden.parquet")

    # Test 1: Sum of deliveries for ANR on 2022-01-01
    plan1 = Plan(
        filters=[
            Filter(column="pipeline_name", op="=", value="ANR Pipeline Company"),
            Filter(column="rec_del_sign", op="=", value=-1),
            Filter(
                column="eff_gas_day", op="between", value=[pl.date(2022, 1, 1), pl.date(2022, 1, 1)]
            ),
        ],
        aggregate=Aggregate(metrics=[{"col": "scheduled_quantity", "fn": "sum"}]),
    )

    result1 = run(lf, plan1)

    # Verify we get a result
    assert isinstance(result1, Answer)
    assert result1.table.height == 1
    assert "sum_scheduled_quantity" in result1.table.columns
    assert result1.evidence["rows_out"] >= 0

    # Test 2: Top 5 states by total scheduled quantity in 2022
    plan2 = Plan(
        filters=[
            Filter(
                column="eff_gas_day",
                op="between",
                value=[pl.date(2022, 1, 1), pl.date(2022, 12, 31)],
            )
        ],
        aggregate=Aggregate(
            groupby=["state_abb"], metrics=[{"col": "scheduled_quantity", "fn": "sum"}]
        ),
        sort=Sort(by=["sum_scheduled_quantity"], desc=True, limit=5),
    )

    result2 = run(lf, plan2)

    # Verify we get results
    assert isinstance(result2, Answer)
    assert result2.table.height <= 5
    assert "state_abb" in result2.table.columns
    assert "sum_scheduled_quantity" in result2.table.columns
    assert result2.evidence["rows_out"] >= 0


def test_evidence_card_structure():
    """Test that evidence card has expected structure."""
    data = {"pipeline_name": ["ANR", "TGP"], "scheduled_quantity": [100.0, 200.0]}
    df = pl.DataFrame(data)
    lf = df.lazy()

    plan = Plan(
        filters=[Filter(column="pipeline_name", op="=", value="ANR")],
        aggregate=Aggregate(metrics=[{"col": "scheduled_quantity", "fn": "sum"}]),
    )

    result = run(lf, plan)
    evidence = result.evidence

    # Check required fields
    required_fields = [
        "filters",
        "aggregate",
        "sort",
        "rows_out",
        "columns",
        "missingness",
        "timings_ms",
        "cache",
        "repro",
    ]

    for field in required_fields:
        assert field in evidence, f"Missing field: {field}"

    # Check specific values
    assert evidence["rows_out"] == 1
    assert "sum_scheduled_quantity" in evidence["columns"]
    assert "plan" in evidence["timings_ms"]
    assert "collect" in evidence["timings_ms"]
    assert evidence["cache"]["hit"] is False
    assert "import polars as pl" in evidence["repro"]["snippet"]
