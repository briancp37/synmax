"""Tests for DAG Plan Schema."""

import pytest
from pydantic import ValidationError

from data_agent.core.agent_schema import (
    AggregateParams,
    ChangepointParams,
    Edge,
    EvidenceCollectParams,
    FilterParams,
    LimitParams,
    PlanGraph,
    RankParams,
    ResampleParams,
    SaveArtifactParams,
    Step,
    StepType,
    STLDeseasonalizeParams,
)


class TestStepType:
    """Test StepType enum."""

    def test_step_type_values(self):
        """Test that all expected step types are available."""
        expected_types = {
            "filter",
            "resample",
            "aggregate",
            "stl_deseasonalize",
            "changepoint",
            "rank",
            "limit",
            "save_artifact",
            "evidence_collect",
        }
        actual_types = {step_type.value for step_type in StepType}
        assert actual_types == expected_types


class TestFilterParams:
    """Test FilterParams validation."""

    def test_valid_filter_params(self):
        """Test valid filter parameter combinations."""
        # Equality filter
        params = FilterParams(column="temperature", op="=", value=25.0)
        assert params.column == "temperature"
        assert params.op == "="
        assert params.value == 25.0

        # In filter
        params = FilterParams(column="status", op="in", value=["active", "pending"])
        assert params.value == ["active", "pending"]

        # Between filter
        params = FilterParams(column="pressure", op="between", value=[10, 20])
        assert params.value == [10, 20]

        # Is not null filter
        params = FilterParams(column="field", op="is_not_null")
        assert params.value is None

    def test_invalid_filter_params(self):
        """Test invalid filter parameter combinations."""
        # is_not_null with value
        with pytest.raises(ValidationError, match="is_not_null operation should not have a value"):
            FilterParams(column="field", op="is_not_null", value="something")

        # in operation without list
        with pytest.raises(ValidationError, match="in operation requires a list value"):
            FilterParams(column="field", op="in", value="not_a_list")

        # between operation without exactly 2 values
        with pytest.raises(
            ValidationError, match="between operation requires a list with exactly 2 values"
        ):
            FilterParams(column="field", op="between", value=[1, 2, 3])

        # equality operation without value
        with pytest.raises(ValidationError, match="= operation requires a value"):
            FilterParams(column="field", op="=", value=None)


class TestAggregateParams:
    """Test AggregateParams validation."""

    def test_valid_aggregate_params(self):
        """Test valid aggregate parameters."""
        params = AggregateParams(
            groupby=["station", "date"],
            metrics=[
                {"col": "flow_rate", "fn": "avg"},
                {"col": "pressure", "fn": "p95"},
                {"col": "id", "fn": "count"},
            ],
        )
        assert params.groupby == ["station", "date"]
        assert len(params.metrics) == 3

    def test_invalid_aggregate_params(self):
        """Test invalid aggregate parameters."""
        # Missing required metric fields
        with pytest.raises(ValidationError, match="Each metric must have 'col' and 'fn' keys"):
            AggregateParams(groupby=["station"], metrics=[{"col": "flow_rate"}])  # Missing 'fn'

        # Invalid function
        with pytest.raises(ValidationError, match="Function 'invalid' not in allowed"):
            AggregateParams(groupby=["station"], metrics=[{"col": "flow_rate", "fn": "invalid"}])


class TestStep:
    """Test Step model."""

    def test_valid_step(self):
        """Test valid step creation."""
        step = Step(
            id="filter_1",
            op=StepType.FILTER,
            params={"column": "temperature", "op": "=", "value": 25.0},
            engine="polars",
        )
        assert step.id == "filter_1"
        assert step.op == StepType.FILTER
        assert step.engine == "polars"

    def test_invalid_step_id(self):
        """Test invalid step ID validation."""
        # Empty ID
        with pytest.raises(ValidationError, match="Step ID must be non-empty"):
            Step(id="", op=StepType.FILTER)

        # Invalid characters
        with pytest.raises(ValidationError, match="Step ID must be non-empty and contain only"):
            Step(id="step with spaces", op=StepType.FILTER)

    def test_step_param_validation(self):
        """Test that step parameters are validated against operation type."""
        # Valid filter parameters
        step = Step(
            id="filter_1", op=StepType.FILTER, params={"column": "temp", "op": "=", "value": 25}
        )
        assert step.params["column"] == "temp"

        # Invalid filter parameters - missing value for equality operation
        with pytest.raises(
            ValidationError, match="Invalid parameters for StepType.FILTER operation"
        ):
            Step(
                id="filter_1",
                op=StepType.FILTER,
                params={"column": "temp", "op": "=", "value": None},  # None value for equality
            )


class TestEdge:
    """Test Edge model."""

    def test_valid_edge(self):
        """Test valid edge creation."""
        edge = Edge(src="step_1", dst="step_2")
        assert edge.src == "step_1"
        assert edge.dst == "step_2"

    def test_invalid_edge_ids(self):
        """Test invalid edge ID validation."""
        # Empty source
        with pytest.raises(ValidationError, match="Node ID must be non-empty"):
            Edge(src="", dst="step_2")

        # Invalid destination
        with pytest.raises(ValidationError, match="Node ID must be non-empty and contain only"):
            Edge(src="step_1", dst="step with spaces")


class TestPlanGraph:
    """Test PlanGraph model and validation."""

    def test_valid_dag(self):
        """Test valid DAG creation."""
        steps = [
            Step(
                id="filter_1", op=StepType.FILTER, params={"column": "temp", "op": "=", "value": 25}
            ),
            Step(
                id="agg_1",
                op=StepType.AGGREGATE,
                params={"groupby": ["station"], "metrics": [{"col": "flow", "fn": "avg"}]},
            ),
        ]
        edges = [Edge(src="raw", dst="filter_1"), Edge(src="filter_1", dst="agg_1")]

        dag = PlanGraph(nodes=steps, edges=edges, outputs=["agg_1"])
        assert len(dag.nodes) == 2
        assert len(dag.edges) == 2
        assert dag.outputs == ["agg_1"]

    def test_duplicate_step_ids(self):
        """Test duplicate step ID validation."""
        steps = [
            Step(
                id="step_1", op=StepType.FILTER, params={"column": "temp", "op": "=", "value": 25}
            ),
            Step(id="step_1", op=StepType.LIMIT, params={"n": 100}),  # Duplicate ID
        ]
        edges = [Edge(src="raw", dst="step_1")]

        with pytest.raises(ValidationError, match="Duplicate step IDs found"):
            PlanGraph(nodes=steps, edges=edges)

    def test_unknown_edge_references(self):
        """Test validation of edge references."""
        steps = [
            Step(id="step_1", op=StepType.FILTER, params={"column": "temp", "op": "=", "value": 25})
        ]

        # Unknown source
        edges = [Edge(src="unknown", dst="step_1")]
        with pytest.raises(ValidationError, match="Edge source 'unknown' not found"):
            PlanGraph(nodes=steps, edges=edges)

        # Unknown destination
        edges = [Edge(src="raw", dst="unknown")]
        with pytest.raises(ValidationError, match="Edge destination 'unknown' not found"):
            PlanGraph(nodes=steps, edges=edges)

    def test_unknown_output_reference(self):
        """Test validation of output references."""
        steps = [
            Step(id="step_1", op=StepType.FILTER, params={"column": "temp", "op": "=", "value": 25})
        ]
        edges = [Edge(src="raw", dst="step_1")]

        with pytest.raises(ValidationError, match="Output 'unknown' not found"):
            PlanGraph(nodes=steps, edges=edges, outputs=["unknown"])

    def test_cycle_detection(self):
        """Test cycle detection in DAG."""
        steps = [
            Step(
                id="step_1", op=StepType.FILTER, params={"column": "temp", "op": "=", "value": 25}
            ),
            Step(id="step_2", op=StepType.LIMIT, params={"n": 100}),
            Step(id="step_3", op=StepType.RANK, params={"by": ["temp"], "descending": True}),
        ]

        # Create a cycle: step_1 -> step_2 -> step_3 -> step_1
        edges = [
            Edge(src="raw", dst="step_1"),
            Edge(src="step_1", dst="step_2"),
            Edge(src="step_2", dst="step_3"),
            Edge(src="step_3", dst="step_1"),  # Creates cycle
        ]

        with pytest.raises(ValidationError, match="Cycle detected"):
            PlanGraph(nodes=steps, edges=edges)

    def test_topological_order(self):
        """Test topological ordering of DAG."""
        steps = [
            Step(
                id="filter_1", op=StepType.FILTER, params={"column": "temp", "op": "=", "value": 25}
            ),
            Step(
                id="agg_1",
                op=StepType.AGGREGATE,
                params={"groupby": ["station"], "metrics": [{"col": "flow", "fn": "avg"}]},
            ),
            Step(id="limit_1", op=StepType.LIMIT, params={"n": 100}),
        ]
        edges = [
            Edge(src="raw", dst="filter_1"),
            Edge(src="filter_1", dst="agg_1"),
            Edge(src="agg_1", dst="limit_1"),
        ]

        dag = PlanGraph(nodes=steps, edges=edges, outputs=["limit_1"])
        order = dag.topological_order()

        # Check that dependencies come before dependents
        assert order.index("raw") < order.index("filter_1")
        assert order.index("filter_1") < order.index("agg_1")
        assert order.index("agg_1") < order.index("limit_1")

    def test_plan_hash(self):
        """Test plan hashing for caching."""
        steps = [
            Step(
                id="filter_1", op=StepType.FILTER, params={"column": "temp", "op": "=", "value": 25}
            ),
            Step(
                id="agg_1",
                op=StepType.AGGREGATE,
                params={"groupby": ["station"], "metrics": [{"col": "flow", "fn": "avg"}]},
            ),
        ]
        edges = [Edge(src="raw", dst="filter_1"), Edge(src="filter_1", dst="agg_1")]

        dag1 = PlanGraph(nodes=steps, edges=edges, outputs=["agg_1"])
        dag2 = PlanGraph(nodes=steps, edges=edges, outputs=["agg_1"])

        # Same DAGs should have same hash
        assert dag1.plan_hash() == dag2.plan_hash()

        # Different DAGs should have different hashes
        dag3 = PlanGraph(nodes=steps[:-1], edges=edges[:-1], outputs=["filter_1"])
        assert dag1.plan_hash() != dag3.plan_hash()

    def test_json_schema_export(self):
        """Test JSON schema export for LLM function-calling."""
        dag = PlanGraph(nodes=[], edges=[])
        schema = dag.to_json_schema()

        assert schema["type"] == "object"
        assert "nodes" in schema["properties"]
        assert "edges" in schema["properties"]
        assert schema["properties"]["nodes"]["type"] == "array"
        assert schema["properties"]["edges"]["type"] == "array"

        # Check that step types are included in enum
        step_enum = schema["properties"]["nodes"]["items"]["properties"]["op"]["enum"]
        expected_ops = [op.value for op in StepType]
        assert set(step_enum) == set(expected_ops)


class TestParameterModels:
    """Test individual parameter model validation."""

    def test_resample_params(self):
        """Test ResampleParams validation."""
        params = ResampleParams(freq="1d", on="timestamp")
        assert params.freq == "1d"
        assert params.on == "timestamp"

        # With aggregation
        params = ResampleParams(freq="1h", on="timestamp", agg={"flow": "mean", "pressure": "max"})
        assert params.agg == {"flow": "mean", "pressure": "max"}

    def test_changepoint_params(self):
        """Test ChangepointParams validation."""
        params = ChangepointParams(column="flow_rate")
        assert params.column == "flow_rate"
        assert params.method == "pelt"  # default

        params = ChangepointParams(column="pressure", method="binseg", min_size=5, jump=10)
        assert params.method == "binseg"
        assert params.min_size == 5
        assert params.jump == 10

    def test_rank_params(self):
        """Test RankParams validation."""
        params = RankParams(by=["temperature", "pressure"])
        assert params.by == ["temperature", "pressure"]
        assert params.method == "average"  # default
        assert params.descending is False  # default

        params = RankParams(by=["score"], method="dense", descending=True)
        assert params.method == "dense"
        assert params.descending is True

    def test_limit_params(self):
        """Test LimitParams validation."""
        params = LimitParams(n=100)
        assert params.n == 100
        assert params.offset == 0  # default

        params = LimitParams(n=50, offset=10)
        assert params.n == 50
        assert params.offset == 10

        # Invalid limit
        with pytest.raises(ValidationError, match="greater than 0"):
            LimitParams(n=0)

        # Invalid offset
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            LimitParams(n=100, offset=-1)

    def test_save_artifact_params(self):
        """Test SaveArtifactParams validation."""
        params = SaveArtifactParams(path="/tmp/output.parquet")
        assert params.path == "/tmp/output.parquet"
        assert params.format == "parquet"  # default
        assert params.overwrite is True  # default

        params = SaveArtifactParams(path="/tmp/data.csv", format="csv", overwrite=False)
        assert params.format == "csv"
        assert params.overwrite is False

    def test_evidence_collect_params(self):
        """Test EvidenceCollectParams validation."""
        params = EvidenceCollectParams()
        assert params.columns is None  # default
        assert params.sample_size == 100  # default
        assert params.method == "random"  # default

        params = EvidenceCollectParams(
            columns=["temperature", "pressure"], sample_size=50, method="head"
        )
        assert params.columns == ["temperature", "pressure"]
        assert params.sample_size == 50
        assert params.method == "head"

        # Invalid sample size
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            EvidenceCollectParams(sample_size=0)

    def test_stl_deseasonalize_params(self):
        """Test STLDeseasonalizeParams validation."""
        params = STLDeseasonalizeParams(column="temperature")
        assert params.column == "temperature"
        assert params.period is None  # default

        params = STLDeseasonalizeParams(column="flow", period=24, seasonal=7, trend=None)
        assert params.column == "flow"
        assert params.period == 24
        assert params.seasonal == 7
        assert params.trend is None


class TestComplexDAGs:
    """Test complex DAG scenarios."""

    def test_multi_branch_dag(self):
        """Test DAG with multiple branches."""
        steps = [
            Step(
                id="filter_temp",
                op=StepType.FILTER,
                params={"column": "temp", "op": ">", "value": 20},
            ),
            Step(
                id="filter_pressure",
                op=StepType.FILTER,
                params={"column": "pressure", "op": "<", "value": 100},
            ),
            Step(
                id="agg_temp",
                op=StepType.AGGREGATE,
                params={"groupby": ["station"], "metrics": [{"col": "temp", "fn": "avg"}]},
            ),
            Step(
                id="agg_pressure",
                op=StepType.AGGREGATE,
                params={"groupby": ["station"], "metrics": [{"col": "pressure", "fn": "avg"}]},
            ),
            Step(id="final_limit", op=StepType.LIMIT, params={"n": 100}),
        ]

        edges = [
            Edge(src="raw", dst="filter_temp"),
            Edge(src="raw", dst="filter_pressure"),
            Edge(src="filter_temp", dst="agg_temp"),
            Edge(src="filter_pressure", dst="agg_pressure"),
            Edge(src="agg_temp", dst="final_limit"),
            Edge(src="agg_pressure", dst="final_limit"),
        ]

        dag = PlanGraph(nodes=steps, edges=edges, outputs=["final_limit"])

        # Should validate successfully
        assert len(dag.nodes) == 5
        assert len(dag.edges) == 6

        # Check topological order respects dependencies
        order = dag.topological_order()
        assert order.index("raw") < order.index("filter_temp")
        assert order.index("raw") < order.index("filter_pressure")
        assert order.index("filter_temp") < order.index("agg_temp")
        assert order.index("filter_pressure") < order.index("agg_pressure")
        assert order.index("agg_temp") < order.index("final_limit")
        assert order.index("agg_pressure") < order.index("final_limit")

    def test_linear_pipeline(self):
        """Test simple linear pipeline."""
        steps = [
            Step(
                id="filter_1",
                op=StepType.FILTER,
                params={"column": "status", "op": "=", "value": "active"},
            ),
            Step(id="resample_1", op=StepType.RESAMPLE, params={"freq": "1d", "on": "timestamp"}),
            Step(
                id="agg_1",
                op=StepType.AGGREGATE,
                params={"groupby": ["date"], "metrics": [{"col": "flow", "fn": "sum"}]},
            ),
            Step(id="rank_1", op=StepType.RANK, params={"by": ["flow"], "descending": True}),
            Step(id="limit_1", op=StepType.LIMIT, params={"n": 10}),
            Step(id="save_1", op=StepType.SAVE_ARTIFACT, params={"path": "output.parquet"}),
        ]

        edges = [
            Edge(src="raw", dst="filter_1"),
            Edge(src="filter_1", dst="resample_1"),
            Edge(src="resample_1", dst="agg_1"),
            Edge(src="agg_1", dst="rank_1"),
            Edge(src="rank_1", dst="limit_1"),
            Edge(src="limit_1", dst="save_1"),
        ]

        dag = PlanGraph(nodes=steps, edges=edges, outputs=["save_1"])

        # Should validate successfully
        assert len(dag.nodes) == 6

        # Check topological order is linear
        order = dag.topological_order()
        expected_order = ["raw", "filter_1", "resample_1", "agg_1", "rank_1", "limit_1", "save_1"]
        for i in range(len(expected_order) - 1):
            assert order.index(expected_order[i]) < order.index(expected_order[i + 1])
