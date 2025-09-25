"""Agent Planner - LLM-driven DAG generation with validation and repair.

This module handles the planning phase of the data agent, where natural language
queries are converted into structured DAG plans using LLMs, with validation
and repair capabilities.
"""

import json
import logging
from typing import Any, Optional

from .agent_schema import Edge, PlanGraph, Step, StepType
from .dsl_loader import create_macro_plan, get_available_macros
from .llm_client import LLMClient, get_default_llm_client

logger = logging.getLogger(__name__)


class PlanValidationError(Exception):
    """Raised when a plan fails validation."""

    pass


class AgentPlanner:
    """Agent planner that converts natural language to DAG plans."""

    def __init__(self, client: Optional[LLMClient] = None):
        """Initialize the planner with an LLM client."""
        self.client = client

    def plan(
        self,
        query: str,
        available_columns: list[str],
        fallback: bool = True,
        temperature: float = 0.1,
    ) -> PlanGraph:
        """Plan a query using LLM with validation and repair.

        Args:
            query: Natural language query
            available_columns: List of available columns in the dataset
            fallback: Whether to use deterministic fallbacks on failure
            temperature: LLM temperature for stability

        Returns:
            Validated and potentially repaired PlanGraph

        Raises:
            PlanValidationError: If plan cannot be validated or repaired
        """
        try:
            # Try LLM planning first
            plan = self._plan_from_llm(query, available_columns, temperature)
            logger.info("LLM planning succeeded", extra={"query": query})
        except Exception as e:
            logger.warning("LLM planning failed", extra={"error": str(e), "query": query})
            if fallback:
                plan = self._fallback_plan(query, available_columns)
                logger.info("Using fallback plan", extra={"query": query})
            else:
                raise PlanValidationError(f"LLM planning failed: {e}") from e

        # Validate and repair the plan
        repaired_plan = self._repair_plan(plan, available_columns)

        # Final validation
        try:
            repaired_plan.model_validate(repaired_plan.model_dump())
        except Exception as e:
            raise PlanValidationError(f"Final validation failed: {e}") from e

        return repaired_plan

    def _plan_from_llm(
        self, query: str, available_columns: list[str], temperature: float
    ) -> PlanGraph:
        """Generate plan using LLM function calling."""
        if self.client is None:
            raise ValueError("No LLM client available for planning")
            
        # Build the function schema
        function_schema = self._build_function_schema(available_columns)

        # Create the prompt
        system_prompt = self._build_system_prompt(available_columns)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a DAG plan for this query: {query}"},
        ]

        # Call LLM with function schema
        response = self.client.call(
            messages=messages,
            tools=[{"type": "function", "function": function_schema}],
            tool_choice={"type": "function", "function": {"name": "create_dag_plan"}},
            temperature=temperature,
        )

        # Extract and parse the function call
        if not response.get("tool_calls"):
            raise ValueError("LLM did not return a function call")

        tool_call = response["tool_calls"][0]
        if tool_call.function.name != "create_dag_plan":
            raise ValueError(f"Unexpected function call: {tool_call.function.name}")

        try:
            plan_data = json.loads(tool_call.function.arguments)
            return PlanGraph(**plan_data)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}") from e

    def _build_function_schema(self, available_columns: list[str]) -> dict[str, Any]:
        """Build the function schema for LLM function calling."""
        # Get the base schema from PlanGraph
        # base_schema = PlanGraph.model_json_schema()  # TODO: Use for enhanced schema

        # Enhance with available columns information
        function_schema = {
            "name": "create_dag_plan",
            "description": "Create a DAG plan for data processing",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "description": "List of processing steps in the DAG",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique identifier for this step "
                                    "(use short names like 'f', 'a', 's')",
                                },
                                "op": {
                                    "type": "string",
                                    "enum": [op.value for op in StepType],
                                    "description": "Operation type",
                                },
                                "params": {
                                    "type": "object",
                                    "description": "Parameters for the operation",
                                },
                                "engine": {
                                    "type": "string",
                                    "enum": ["polars", "duckdb"],
                                    "default": "polars",
                                    "description": "Processing engine to use",
                                },
                                "materialize": {
                                    "type": "boolean",
                                    "description": "Whether to materialize intermediate results",
                                },
                            },
                            "required": ["id", "op", "params"],
                        },
                    },
                    "edges": {
                        "type": "array",
                        "description": "Connections between steps",
                        "items": {
                            "type": "object",
                            "properties": {
                                "src": {"type": "string", "description": "Source step ID"},
                                "dst": {"type": "string", "description": "Destination step ID"},
                            },
                            "required": ["src", "dst"],
                        },
                    },
                    "inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["raw"],
                        "description": "Input data sources",
                    },
                    "outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Output step IDs",
                    },
                },
                "required": ["nodes", "edges"],
            },
        }

        return function_schema

    def _build_system_prompt(self, available_columns: list[str]) -> str:
        """Build the system prompt for LLM planning."""
        columns_str = ", ".join(available_columns)

        example_plan = {
            "nodes": [
                {
                    "id": "f",
                    "op": "filter",
                    "params": {
                        "column": "eff_gas_day",
                        "op": "between",
                        "value": ["2022-01-01", "2022-12-31"],
                    },
                },
                {
                    "id": "a",
                    "op": "aggregate",
                    "params": {
                        "groupby": ["pipeline_name", "eff_gas_day"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}],
                    },
                },
                {"id": "s", "op": "stl_deseasonalize", "params": {"column": "sum_scheduled_quantity"}},
                {
                    "id": "c",
                    "op": "changepoint",
                    "params": {
                        "column": "deseasonalized", 
                        "method": "pelt", 
                        "min_size": 7,
                        "penalty": 1.0,
                        "min_confidence": 0.1,
                        "groupby": ["pipeline_name"]
                    },
                },
                {"id": "r", "op": "rank", "params": {"by": ["change_magnitude"], "descending": True}},
                {"id": "l", "op": "limit", "params": {"n": 10}},
                {"id": "e", "op": "evidence_collect", "params": {}},
            ],
            "edges": [
                {"src": "raw", "dst": "f"},
                {"src": "f", "dst": "a"},
                {"src": "a", "dst": "s"},
                {"src": "s", "dst": "c"},
                {"src": "c", "dst": "r"},
                {"src": "r", "dst": "l"},
                {"src": "l", "dst": "e"},
            ],
            "inputs": ["raw"],
            "outputs": ["l"],
        }

        return f"""You are an expert data processing planner. 
Convert natural language queries into structured DAG plans for gas pipeline data analysis.

Available columns: {columns_str}

Key data semantics:
- eff_gas_day: Date column (YYYY-MM-DD format)
- scheduled_quantity: Flow quantity (positive values)
- rec_del_sign: +1 for receipts, -1 for deliveries
- pipeline_name: Pipeline identifier
- state_abb: Two-letter state code
- category_short: Entity category (e.g., Interconnect, LDC, Industrial)

Available operations:
- filter: Apply conditions (column, op, value). 
  Ops: =, !=, in, between, is_not_null, contains, >, <, >=, <=
- resample: Resample time series (freq, on, agg)
- aggregate: Group and summarize (groupby, metrics). 
  Metrics fn: sum, count, avg, p95, p50, min, max, std
- stl_deseasonalize: Remove seasonal patterns (column, period, seasonal, trend)
- changepoint: Detect regime changes (column, method, min_size, jump)
- rank: Rank results (by, method, descending)
- limit: Limit results (n, offset)
- save_artifact: Save to file (path, format, overwrite)
- evidence_collect: Collect execution evidence (columns, sample_size, method)

Guidelines:
1. Start with "raw" as input source
2. Use short step IDs (f, a, s, c, r, l, e)
3. Create linear pipelines typically: filter → aggregate → analysis → rank → limit → evidence
4. Always end with evidence_collect unless saving artifacts
5. Use between for date ranges: ["2022-01-01", "2022-12-31"] (dataset is 2022+)
6. For time series analysis: aggregate first, then deseasonalize, then changepoint

CRITICAL - Column naming rules:
- After aggregate with sum: column becomes "sum_<original_column>"
- After stl_deseasonalize: creates "deseasonalized" column  
- After changepoint: creates "change_magnitude" column (not "abs_delta_mean")
- Always include eff_gas_day in groupby for time series: ["pipeline_name", "eff_gas_day"]
- Changepoint needs groupby: ["pipeline_name"] for per-pipeline analysis
- For changepoint detection, use: penalty: 1.0, min_confidence: 0.1 (more sensitive)

7. Set outputs to the final meaningful step (usually before evidence_collect)

Example plan for "regime shifts in 2021 after seasonality removal":
{json.dumps(example_plan, indent=2)}

Create a DAG plan that efficiently answers the user's query."""

    def _fallback_plan(self, query: str, available_columns: list[str]) -> PlanGraph:
        """Generate a deterministic fallback plan for common query patterns."""
        query_lower = query.lower()

        # Time series patterns (check first - most specific)
        if any(word in query_lower for word in ["trend", "change", "shift", "regime", "changepoint", "structural", "break", "seasonality", "deseasonalize"]):
            logger.info("Using time_series_analysis macro for fallback")
            return create_macro_plan("time_series_analysis")

        # Top-k patterns
        if any(word in query_lower for word in ["top", "highest", "largest", "biggest"]):
            # Extract number if present
            import re
            numbers = re.findall(r"\d+", query)
            k = int(numbers[0]) if numbers else 10
            logger.info(f"Using top_k_ranking macro for fallback with k={k}")
            return create_macro_plan("top_k_ranking", k=k)

        # Simple aggregation patterns
        if any(word in query_lower for word in ["sum", "total", "aggregate"]):
            logger.info("Using simple_aggregation macro for fallback")
            return create_macro_plan("simple_aggregation")

        # Default simple plan - use simple aggregation macro
        logger.info("Using default simple_aggregation macro for fallback")
        return create_macro_plan("simple_aggregation", limit=100)





    def _repair_plan(self, plan: PlanGraph, available_columns: list[str]) -> PlanGraph:
        """Repair and validate a plan."""
        repaired_plan = plan.model_copy(deep=True)
        repair_log = []

        # Ensure there's at least one source → sink path
        if not self._has_valid_path(repaired_plan):
            repair_log.append("No valid source-to-sink path found")
            repaired_plan = self._add_missing_connections(repaired_plan)

        # Ensure evidence_collect is present if outputs are specified
        if repaired_plan.outputs and not self._has_evidence_collect(repaired_plan):
            repair_log.append("Added missing evidence_collect step")
            repaired_plan = self._add_evidence_collect(repaired_plan)

        # Insert limit if result might explode
        if self._needs_limit(repaired_plan):
            repair_log.append("Added limit to prevent result explosion")
            repaired_plan = self._add_limit(repaired_plan)

        # Validate column references
        repaired_plan = self._validate_column_references(
            repaired_plan, available_columns, repair_log
        )

        if repair_log:
            logger.info("Plan repaired", extra={"repairs": repair_log})

        return repaired_plan

    def plan_from_macro(self, macro_name: str, **kwargs: Any) -> PlanGraph:
        """Create a plan using a predefined macro.
        
        Args:
            macro_name: Name of the macro to use
            **kwargs: Additional parameters for the macro
            
        Returns:
            PlanGraph created from the macro
            
        Raises:
            ValueError: If the macro name is not recognized
        """
        try:
            plan = create_macro_plan(macro_name, **kwargs)
            logger.info(f"Created plan from macro: {macro_name}")
            return plan
        except ValueError as e:
            available_macros = list(get_available_macros().keys())
            raise ValueError(f"Unknown macro '{macro_name}'. Available macros: {available_macros}") from e

    def _has_valid_path(self, plan: PlanGraph) -> bool:
        """Check if there's a valid path from inputs to outputs."""
        if not plan.outputs:
            return True  # No outputs specified, any structure is valid

        # Build adjacency list
        graph = {}
        all_nodes = set([step.id for step in plan.nodes] + plan.inputs)
        for node in all_nodes:
            graph[node] = []

        for edge in plan.edges:
            graph[edge.src].append(edge.dst)

        # Check reachability from inputs to outputs
        for output in plan.outputs:
            reachable = False
            for input_node in plan.inputs:
                if self._is_reachable(graph, input_node, output):
                    reachable = True
                    break
            if not reachable:
                return False

        return True

    def _is_reachable(self, graph: dict[str, list[str]], start: str, end: str) -> bool:
        """Check if end is reachable from start using BFS."""
        if start == end:
            return True

        visited = set()
        queue = [start]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            for neighbor in graph.get(node, []):
                if neighbor == end:
                    return True
                if neighbor not in visited:
                    queue.append(neighbor)

        return False

    def _add_missing_connections(self, plan: PlanGraph) -> PlanGraph:
        """Add missing connections to create valid paths."""
        # Simple repair: connect raw to first node if no connections exist
        if not plan.edges and plan.nodes:
            first_node = plan.nodes[0]
            plan.edges.append(Edge(src="raw", dst=first_node.id))
            if not plan.outputs:
                plan.outputs = [first_node.id]

        return plan

    def _has_evidence_collect(self, plan: PlanGraph) -> bool:
        """Check if plan has evidence_collect step."""
        return any(step.op == StepType.EVIDENCE_COLLECT for step in plan.nodes)

    def _add_evidence_collect(self, plan: PlanGraph) -> PlanGraph:
        """Add evidence_collect step to the plan."""
        evidence_step = Step(id="evidence", op=StepType.EVIDENCE_COLLECT, params={})
        plan.nodes.append(evidence_step)

        # Connect final outputs to evidence
        if plan.outputs:
            for output_id in plan.outputs:
                plan.edges.append(Edge(src=output_id, dst="evidence"))

        return plan

    def _needs_limit(self, plan: PlanGraph) -> bool:
        """Check if plan needs a limit to prevent result explosion."""
        # Look for aggregations without limits
        has_aggregation = any(step.op == StepType.AGGREGATE for step in plan.nodes)
        has_limit = any(step.op == StepType.LIMIT for step in plan.nodes)

        return has_aggregation and not has_limit

    def _add_limit(self, plan: PlanGraph) -> PlanGraph:
        """Add a limit step to prevent result explosion."""
        # Find the last meaningful step before evidence_collect
        evidence_steps = [s for s in plan.nodes if s.op == StepType.EVIDENCE_COLLECT]
        if evidence_steps:
            evidence_step = evidence_steps[0]
            # Find steps that connect to evidence
            incoming_edges = [e for e in plan.edges if e.dst == evidence_step.id]
            if incoming_edges:
                # Insert limit before evidence
                limit_step = Step(id="auto_limit", op=StepType.LIMIT, params={"n": 1000})
                plan.nodes.append(limit_step)

                # Redirect edges
                for edge in incoming_edges:
                    edge.dst = limit_step.id
                plan.edges.append(Edge(src=limit_step.id, dst=evidence_step.id))

        return plan

    def _validate_column_references(
        self, plan: PlanGraph, available_columns: list[str], repair_log: list[str]
    ) -> PlanGraph:
        """Validate and repair column references in the plan."""
        available_set = set(available_columns)

        for step in plan.nodes:
            if step.op == StepType.FILTER:
                column = step.params.get("column")
                if column and column not in available_set:
                    repair_log.append(f"Invalid column '{column}' in filter step '{step.id}'")
                    # Try to find a similar column
                    similar = self._find_similar_column(column, available_columns)
                    if similar:
                        step.params["column"] = similar
                        repair_log.append(f"Replaced with similar column '{similar}'")

            elif step.op == StepType.AGGREGATE:
                groupby = step.params.get("groupby", [])
                for i, col in enumerate(groupby):
                    if col not in available_set:
                        similar = self._find_similar_column(col, available_columns)
                        if similar:
                            groupby[i] = similar
                            repair_log.append(f"Replaced groupby column '{col}' with '{similar}'")

                metrics = step.params.get("metrics", [])
                for metric in metrics:
                    col = metric.get("col")
                    if col and col not in available_set:
                        similar = self._find_similar_column(col, available_columns)
                        if similar:
                            metric["col"] = similar
                            repair_log.append(f"Replaced metric column '{col}' with '{similar}'")

        return plan

    def _find_similar_column(self, target: str, available: list[str]) -> Optional[str]:
        """Find the most similar column name."""
        target_lower = target.lower()

        # Exact match (case insensitive)
        for col in available:
            if col.lower() == target_lower:
                return col

        # Substring match
        for col in available:
            if target_lower in col.lower() or col.lower() in target_lower:
                return col

        return None


def plan_from_llm(query: str, client: Optional[LLMClient] = None) -> PlanGraph:
    """Convenience function for LLM-based planning."""
    if client is None:
        client = get_default_llm_client()
    planner = AgentPlanner(client)
    # For now, use a basic set of columns - in real usage this would come from data dictionary
    available_columns = [
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
    return planner.plan(query, available_columns)


def estimate_plan_complexity(plan: PlanGraph) -> dict[str, Any]:
    """Estimate the complexity and resource requirements of a plan."""
    estimates = {
        "steps": len(plan.nodes),
        "edges": len(plan.edges),
        "estimated_time_seconds": 0,
        "estimated_memory_mb": 0,
        "will_checkpoint": [],
        "topological_order": plan.topological_order(),
    }

    # Simple heuristics for estimation
    for step in plan.nodes:
        if step.op in [StepType.STL_DESEASONALIZE, StepType.CHANGEPOINT]:
            estimates["estimated_time_seconds"] += 5
            estimates["estimated_memory_mb"] += 100
            estimates["will_checkpoint"].append(step.id)
        elif step.op == StepType.AGGREGATE:
            estimates["estimated_time_seconds"] += 2
            estimates["estimated_memory_mb"] += 50
        else:
            estimates["estimated_time_seconds"] += 0.5
            estimates["estimated_memory_mb"] += 10

    return estimates
