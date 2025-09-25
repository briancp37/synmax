"""DAG Plan Schema for SynMax Data Agent.

This module defines the typed DAG structure for data processing plans,
including steps, edges, and validation logic.
"""

import hashlib
import json
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class StepType(str, Enum):
    """Supported step types in the DAG."""

    FILTER = "filter"
    RESAMPLE = "resample"
    AGGREGATE = "aggregate"
    STL_DESEASONALIZE = "stl_deseasonalize"
    CHANGEPOINT = "changepoint"
    RANK = "rank"
    LIMIT = "limit"
    SAVE_ARTIFACT = "save_artifact"
    EVIDENCE_COLLECT = "evidence_collect"


class FilterParams(BaseModel):
    """Parameters for filter operation."""

    column: str
    op: Literal["=", "in", "between", "is_not_null", "contains", "!=", ">", "<", ">=", "<="]
    value: Optional[Union[str, int, float, list[Union[str, int, float]]]] = None

    @field_validator("value")
    @classmethod
    def validate_value_for_op(cls, v: Any, info: Any) -> Any:
        """Validate value matches operation requirements."""
        if not hasattr(info, "data") or "op" not in info.data:
            return v

        op = info.data["op"]
        if op == "is_not_null" and v is not None:
            raise ValueError("is_not_null operation should not have a value")
        elif op == "in" and not isinstance(v, list):
            raise ValueError("in operation requires a list value")
        elif op == "between" and (not isinstance(v, list) or len(v) != 2):
            raise ValueError("between operation requires a list with exactly 2 values")
        elif op in ["=", "!=", ">", "<", ">=", "<=", "contains"] and v is None:
            raise ValueError(f"{op} operation requires a value")
        return v


class ResampleParams(BaseModel):
    """Parameters for resample operation."""

    freq: str = Field(description="Frequency string like '1d', '1h', '15min'")
    on: str = Field(description="Column to resample on")
    agg: Optional[dict[str, str]] = Field(
        default=None, description="Aggregation functions per column"
    )


class AggregateParams(BaseModel):
    """Parameters for aggregate operation."""

    groupby: list[str] = Field(description="Columns to group by")
    metrics: list[dict[str, str]] = Field(description="Metrics to compute")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        """Validate metric definitions."""
        allowed_fns = {"sum", "count", "avg", "p95", "p50", "min", "max", "std"}
        for metric in v:
            if "col" not in metric or "fn" not in metric:
                raise ValueError("Each metric must have 'col' and 'fn' keys")
            if metric["fn"] not in allowed_fns:
                raise ValueError(f"Function '{metric['fn']}' not in allowed: {allowed_fns}")
        return v


class STLDeseasonalizeParams(BaseModel):
    """Parameters for STL deseasonalization."""

    column: str = Field(description="Column to deseasonalize")
    period: Optional[int] = Field(default=None, description="Seasonal period")
    seasonal: Optional[int] = Field(default=None, description="Seasonal smoother parameter")
    trend: Optional[int] = Field(default=None, description="Trend smoother parameter")


class ChangepointParams(BaseModel):
    """Parameters for changepoint detection."""

    column: str = Field(description="Column to analyze for changepoints")
    method: Literal["pelt", "binseg", "window"] = Field(default="pelt")
    min_size: int = Field(default=2, description="Minimum segment size")
    jump: int = Field(default=5, description="Jump parameter for optimization")


class RankParams(BaseModel):
    """Parameters for ranking operation."""

    by: list[str] = Field(description="Columns to rank by")
    method: Literal["average", "min", "max", "first", "dense"] = Field(default="average")
    descending: bool = Field(default=False)


class LimitParams(BaseModel):
    """Parameters for limit operation."""

    n: int = Field(gt=0, description="Number of rows to limit to")
    offset: int = Field(default=0, ge=0, description="Number of rows to skip")


class SaveArtifactParams(BaseModel):
    """Parameters for saving artifacts."""

    path: str = Field(description="Path to save artifact")
    format: Literal["parquet", "csv", "json"] = Field(default="parquet")
    overwrite: bool = Field(default=True)


class EvidenceCollectParams(BaseModel):
    """Parameters for evidence collection."""

    columns: Optional[list[str]] = Field(
        default=None, description="Columns to collect evidence for"
    )
    sample_size: int = Field(default=100, ge=1, description="Number of samples to collect")
    method: Literal["random", "head", "tail"] = Field(default="random")


class Step(BaseModel):
    """A single step in the DAG."""

    id: str = Field(description="Unique identifier for this step")
    op: StepType = Field(description="Operation type")
    params: dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    engine: Literal["polars", "duckdb"] = Field(default="polars")
    materialize: Optional[bool] = Field(
        default=None, description="Whether to materialize this step"
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate step ID format."""
        if not v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Step ID must be non-empty and contain only alphanumeric, "
                "underscore, or dash characters"
            )
        return v

    @model_validator(mode="after")
    def validate_params(self) -> "Step":
        """Validate parameters match the operation type."""
        param_classes = {
            StepType.FILTER: FilterParams,
            StepType.RESAMPLE: ResampleParams,
            StepType.AGGREGATE: AggregateParams,
            StepType.STL_DESEASONALIZE: STLDeseasonalizeParams,
            StepType.CHANGEPOINT: ChangepointParams,
            StepType.RANK: RankParams,
            StepType.LIMIT: LimitParams,
            StepType.SAVE_ARTIFACT: SaveArtifactParams,
            StepType.EVIDENCE_COLLECT: EvidenceCollectParams,
        }

        if self.op in param_classes:
            try:
                param_classes[self.op](**self.params)
            except Exception as e:
                raise ValueError(f"Invalid parameters for {self.op} operation: {e}") from e

        return self


class Edge(BaseModel):
    """An edge connecting two steps in the DAG."""

    src: str = Field(description="Source step ID")
    dst: str = Field(description="Destination step ID")

    @field_validator("src", "dst")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        """Validate node ID format."""
        if not v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Node ID must be non-empty and contain only alphanumeric, "
                "underscore, or dash characters"
            )
        return v


class PlanGraph(BaseModel):
    """A complete DAG plan with nodes and edges."""

    nodes: list[Step] = Field(description="List of steps in the DAG")
    edges: list[Edge] = Field(description="List of edges connecting steps")
    inputs: list[str] = Field(default=["raw"], description="Input data sources")
    outputs: list[str] = Field(default_factory=list, description="Output step IDs")

    @model_validator(mode="after")
    def validate_dag(self) -> "PlanGraph":
        """Validate the DAG structure."""
        # Check for unique step IDs
        step_ids = [step.id for step in self.nodes]
        if len(step_ids) != len(set(step_ids)):
            duplicates = [id for id in step_ids if step_ids.count(id) > 1]
            raise ValueError(f"Duplicate step IDs found: {set(duplicates)}")

        # Check that all edge references exist
        all_node_ids = set(step_ids) | set(self.inputs)
        for edge in self.edges:
            if edge.src not in all_node_ids:
                raise ValueError(f"Edge source '{edge.src}' not found in nodes or inputs")
            if edge.dst not in step_ids:
                raise ValueError(f"Edge destination '{edge.dst}' not found in nodes")

        # Check that output references exist
        for output_id in self.outputs:
            if output_id not in step_ids:
                raise ValueError(f"Output '{output_id}' not found in nodes")

        # Check for cycles
        self._check_cycles()

        return self

    def _check_cycles(self) -> None:
        """Check for cycles in the DAG using DFS."""
        # Build adjacency list
        graph: dict[str, list[str]] = {}
        all_nodes = set([step.id for step in self.nodes] + self.inputs)

        for node in all_nodes:
            graph[node] = []

        for edge in self.edges:
            graph[edge.src].append(edge.dst)

        # DFS to detect cycles
        white, gray, black = 0, 1, 2
        colors = {node: white for node in all_nodes}

        def dfs(node: str, path: list[str]) -> None:
            if colors[node] == gray:
                cycle_start = path.index(node)
                cycle = " -> ".join(path[cycle_start:] + [node])
                raise ValueError(f"Cycle detected: {cycle}")

            if colors[node] == black:
                return

            colors[node] = gray
            path.append(node)

            for neighbor in graph[node]:
                dfs(neighbor, path)

            path.pop()
            colors[node] = black

        for node in all_nodes:
            if colors[node] == white:
                dfs(node, [])

    def topological_order(self) -> list[str]:
        """Return nodes in topological order."""
        # Build adjacency list and in-degree count
        graph: dict[str, list[str]] = {}
        in_degree: dict[str, int] = {}
        all_nodes = set([step.id for step in self.nodes] + self.inputs)

        for node in all_nodes:
            graph[node] = []
            in_degree[node] = 0

        for edge in self.edges:
            graph[edge.src].append(edge.dst)
            in_degree[edge.dst] += 1

        # Kahn's algorithm
        queue = [node for node in all_nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(all_nodes):
            raise ValueError("Cannot compute topological order: graph has cycles")

        return result

    def plan_hash(self) -> str:
        """Generate a canonical hash of the plan for caching."""
        # Create canonical representation
        canonical = {
            "nodes": sorted(
                [
                    {
                        "id": step.id,
                        "op": step.op,
                        "params": step.params,
                        "engine": step.engine,
                        "materialize": step.materialize,
                    }
                    for step in self.nodes
                ],
                key=lambda x: str(x["id"]),
            ),
            "edges": sorted(
                [{"src": edge.src, "dst": edge.dst} for edge in self.edges],
                key=lambda x: (x["src"], x["dst"]),
            ),
            "inputs": sorted(self.inputs),
            "outputs": sorted(self.outputs),
        }

        # Convert to canonical JSON and hash
        canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode()).hexdigest()

    def to_json_schema(self) -> dict[str, Any]:
        """Export JSON schema for LLM function-calling."""
        return {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "op": {"type": "string", "enum": [op.value for op in StepType]},
                            "params": {"type": "object"},
                            "engine": {
                                "type": "string",
                                "enum": ["polars", "duckdb"],
                                "default": "polars",
                            },
                            "materialize": {"type": "boolean"},
                        },
                        "required": ["id", "op"],
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
                        "required": ["src", "dst"],
                    },
                },
                "inputs": {"type": "array", "items": {"type": "string"}, "default": ["raw"]},
                "outputs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["nodes", "edges"],
        }
