"""DSL Loader for parsing and validating user-provided DAG JSON files.

This module provides functionality to load, validate, and parse DAG plans from JSON files,
enabling deterministic execution without requiring LLM access.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .agent_schema import PlanGraph, Step, Edge, StepType

logger = logging.getLogger(__name__)


class DSLValidationError(Exception):
    """Raised when DSL validation fails."""
    pass


class DSLLoader:
    """Loader for DAG plans from JSON files with validation."""
    
    def __init__(self):
        """Initialize the DSL loader."""
        pass
    
    def load_plan(self, plan_path: Union[str, Path]) -> PlanGraph:
        """Load and validate a plan from a JSON file.
        
        Args:
            plan_path: Path to the JSON plan file
            
        Returns:
            Validated PlanGraph instance
            
        Raises:
            DSLValidationError: If the plan is invalid
            FileNotFoundError: If the file doesn't exist
        """
        plan_path = Path(plan_path)
        
        if not plan_path.exists():
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        
        try:
            with open(plan_path, 'r') as f:
                plan_data = json.load(f)
        except json.JSONDecodeError as e:
            raise DSLValidationError(f"Invalid JSON in plan file: {e}") from e
        
        # Validate and create PlanGraph
        try:
            plan_graph = PlanGraph(**plan_data)
        except Exception as e:
            raise DSLValidationError(f"Invalid plan structure: {e}") from e
        
        # Additional validation
        self._validate_plan_structure(plan_graph)
        
        logger.info(f"Successfully loaded plan from {plan_path}")
        return plan_graph
    
    def _validate_plan_structure(self, plan: PlanGraph) -> None:
        """Perform additional validation on the plan structure.
        
        Args:
            plan: The plan to validate
            
        Raises:
            DSLValidationError: If validation fails
        """
        # Check that all step IDs are unique
        step_ids = [step.id for step in plan.nodes]
        if len(step_ids) != len(set(step_ids)):
            raise DSLValidationError("Duplicate step IDs found in plan")
        
        # Check that all edges reference existing steps
        all_node_ids = set(step_ids + plan.inputs)
        for edge in plan.edges:
            if edge.src not in all_node_ids:
                raise DSLValidationError(f"Edge references non-existent source step: {edge.src}")
            if edge.dst not in step_ids:
                raise DSLValidationError(f"Edge references non-existent destination step: {edge.dst}")
        
        # Check that outputs reference existing steps
        for output_id in plan.outputs:
            if output_id not in step_ids:
                raise DSLValidationError(f"Output references non-existent step: {output_id}")
        
        # Check for cycles (basic check)
        if self._has_cycle(plan):
            raise DSLValidationError("Plan contains cycles")
        
        # Validate step parameters
        for step in plan.nodes:
            self._validate_step_params(step)
    
    def _has_cycle(self, plan: PlanGraph) -> bool:
        """Check if the plan has cycles using DFS.
        
        Args:
            plan: The plan to check
            
        Returns:
            True if cycles are found, False otherwise
        """
        # Build adjacency list
        graph = {step.id: [] for step in plan.nodes}
        for step_id in plan.inputs:
            graph[step_id] = []
        
        for edge in plan.edges:
            if edge.src in graph:
                graph[edge.src].append(edge.dst)
        
        # DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def dfs(node: str) -> bool:
            if node in rec_stack:
                return True  # Back edge found - cycle detected
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def _validate_step_params(self, step: Step) -> None:
        """Validate parameters for a specific step.
        
        Args:
            step: The step to validate
            
        Raises:
            DSLValidationError: If step parameters are invalid
        """
        if step.op == StepType.FILTER:
            required_params = ['column', 'op']
            for param in required_params:
                if param not in step.params:
                    raise DSLValidationError(f"Filter step {step.id} missing required parameter: {param}")
            
            # Validate filter operation
            valid_ops = ["=", "!=", "in", "between", "is_not_null", "contains", ">", "<", ">=", "<="]
            if step.params['op'] not in valid_ops:
                raise DSLValidationError(f"Invalid filter operation: {step.params['op']}")
        
        elif step.op == StepType.AGGREGATE:
            if 'metrics' not in step.params:
                raise DSLValidationError(f"Aggregate step {step.id} missing required parameter: metrics")
            
            metrics = step.params['metrics']
            if not isinstance(metrics, list) or not metrics:
                raise DSLValidationError(f"Aggregate step {step.id} metrics must be a non-empty list")
            
            for metric in metrics:
                if not isinstance(metric, dict) or 'col' not in metric or 'fn' not in metric:
                    raise DSLValidationError(f"Invalid metric in aggregate step {step.id}")
                
                valid_fns = ['sum', 'count', 'avg', 'p95', 'p50', 'min', 'max', 'std']
                if metric['fn'] not in valid_fns:
                    raise DSLValidationError(f"Invalid aggregation function: {metric['fn']}")
        
        elif step.op == StepType.CHANGEPOINT:
            required_params = ['column']
            for param in required_params:
                if param not in step.params:
                    raise DSLValidationError(f"Changepoint step {step.id} missing required parameter: {param}")
        
        elif step.op == StepType.RANK:
            if 'by' not in step.params:
                raise DSLValidationError(f"Rank step {step.id} missing required parameter: by")
            
            if not isinstance(step.params['by'], list):
                raise DSLValidationError(f"Rank step {step.id} 'by' parameter must be a list")
        
        elif step.op == StepType.LIMIT:
            if 'n' not in step.params:
                raise DSLValidationError(f"Limit step {step.id} missing required parameter: n")
            
            if not isinstance(step.params['n'], int) or step.params['n'] <= 0:
                raise DSLValidationError(f"Limit step {step.id} 'n' parameter must be a positive integer")


def create_macro_plan(macro_name: str, **kwargs: Any) -> PlanGraph:
    """Create a predefined DAG macro plan.
    
    Args:
        macro_name: Name of the macro to create
        **kwargs: Additional parameters for the macro
        
    Returns:
        PlanGraph for the requested macro
        
    Raises:
        ValueError: If the macro name is not recognized
    """
    if macro_name == "simple_aggregation":
        return _create_simple_aggregation_macro(**kwargs)
    elif macro_name == "time_series_analysis":
        return _create_time_series_analysis_macro(**kwargs)
    elif macro_name == "top_k_ranking":
        return _create_top_k_ranking_macro(**kwargs)
    else:
        raise ValueError(f"Unknown macro: {macro_name}")


def _create_simple_aggregation_macro(
    groupby_cols: Optional[List[str]] = None,
    metric_col: str = "scheduled_quantity",
    metric_fn: str = "sum",
    limit: int = 100
) -> PlanGraph:
    """Create a simple aggregation macro.
    
    Args:
        groupby_cols: Columns to group by
        metric_col: Column to aggregate
        metric_fn: Aggregation function
        limit: Number of results to return
        
    Returns:
        PlanGraph for simple aggregation
    """
    groupby_cols = groupby_cols or ["pipeline_name"]
    
    return PlanGraph(
        nodes=[
            Step(
                id="a",
                op=StepType.AGGREGATE,
                params={
                    "groupby": groupby_cols,
                    "metrics": [{"col": metric_col, "fn": metric_fn}],
                },
            ),
            Step(id="l", op=StepType.LIMIT, params={"n": limit}),
            Step(id="e", op=StepType.EVIDENCE_COLLECT, params={}),
        ],
        edges=[
            Edge(src="raw", dst="a"),
            Edge(src="a", dst="l"),
            Edge(src="l", dst="e"),
        ],
        inputs=["raw"],
        outputs=["l"],
    )


def _create_time_series_analysis_macro(
    date_range: Optional[List[str]] = None,
    groupby_cols: Optional[List[str]] = None,
    value_col: str = "scheduled_quantity",
    limit: int = 10
) -> PlanGraph:
    """Create a time series analysis macro (STL → changepoint → rank).
    
    Args:
        date_range: Date range filter [start, end]
        groupby_cols: Columns to group by
        value_col: Column to analyze
        limit: Number of results to return
        
    Returns:
        PlanGraph for time series analysis
    """
    date_range = date_range or ["2022-01-01", "2022-12-31"]
    groupby_cols = groupby_cols or ["pipeline_name"]
    
    nodes = []
    edges = []
    
    # Add date filter if specified
    if date_range:
        nodes.append(Step(
            id="f",
            op=StepType.FILTER,
            params={
                "column": "eff_gas_day",
                "op": "between",
                "value": date_range,
            },
        ))
        edges.append(Edge(src="raw", dst="f"))
        prev_step = "f"
    else:
        prev_step = "raw"
    
    # Aggregate step
    nodes.extend([
        Step(
            id="a",
            op=StepType.AGGREGATE,
            params={
                "groupby": groupby_cols + ["eff_gas_day"],
                "metrics": [{"col": value_col, "fn": "sum"}],
            },
        ),
        Step(
            id="s",
            op=StepType.STL_DESEASONALIZE,
            params={"column": f"sum_{value_col}"},
        ),
        Step(
            id="c",
            op=StepType.CHANGEPOINT,
            params={
                "column": "deseasonalized",
                "method": "pelt",
                "min_size": 7,
                "penalty": 1.0,
                "min_confidence": 0.1,
                "groupby": groupby_cols,
            },
        ),
        Step(
            id="r",
            op=StepType.RANK,
            params={"by": ["change_magnitude"], "descending": True},
        ),
        Step(id="l", op=StepType.LIMIT, params={"n": limit}),
        Step(id="e", op=StepType.EVIDENCE_COLLECT, params={}),
    ])
    
    edges.extend([
        Edge(src=prev_step, dst="a"),
        Edge(src="a", dst="s"),
        Edge(src="s", dst="c"),
        Edge(src="c", dst="r"),
        Edge(src="r", dst="l"),
        Edge(src="l", dst="e"),
    ])
    
    return PlanGraph(
        nodes=nodes,
        edges=edges,
        inputs=["raw"],
        outputs=["l"],
    )


def _create_top_k_ranking_macro(
    groupby_cols: Optional[List[str]] = None,
    metric_col: str = "scheduled_quantity",
    metric_fn: str = "sum",
    k: int = 10,
    descending: bool = True
) -> PlanGraph:
    """Create a top-k ranking macro.
    
    Args:
        groupby_cols: Columns to group by
        metric_col: Column to rank by
        metric_fn: Aggregation function
        k: Number of top results
        descending: Whether to rank in descending order
        
    Returns:
        PlanGraph for top-k ranking
    """
    groupby_cols = groupby_cols or ["pipeline_name"]
    
    return PlanGraph(
        nodes=[
            Step(
                id="a",
                op=StepType.AGGREGATE,
                params={
                    "groupby": groupby_cols,
                    "metrics": [{"col": metric_col, "fn": metric_fn}],
                },
            ),
            Step(
                id="r",
                op=StepType.RANK,
                params={"by": [f"{metric_fn}_{metric_col}"], "descending": descending},
            ),
            Step(id="l", op=StepType.LIMIT, params={"n": k}),
            Step(id="e", op=StepType.EVIDENCE_COLLECT, params={}),
        ],
        edges=[
            Edge(src="raw", dst="a"),
            Edge(src="a", dst="r"),
            Edge(src="r", dst="l"),
            Edge(src="l", dst="e"),
        ],
        inputs=["raw"],
        outputs=["l"],
    )


def get_available_macros() -> Dict[str, str]:
    """Get list of available macro plans.
    
    Returns:
        Dictionary mapping macro names to descriptions
    """
    return {
        "simple_aggregation": "Basic aggregation with grouping and limiting",
        "time_series_analysis": "STL deseasonalization → changepoint detection → ranking",
        "top_k_ranking": "Aggregate and rank to find top-k results",
    }
