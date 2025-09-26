"""Agent Shim - Compile Legacy CLI to Tiny DAGs

This module provides a compatibility layer that takes legacy CLI command shapes
and compiles them into tiny DAGs for execution through the new agent executor.
No legacy executor is retained; all paths flow through the DAG engine.
"""

from __future__ import annotations
from typing import Any
from .agent_schema import PlanGraph, Step, Edge


def _tiny_sum_by_group(group: list[str], date_range: tuple[str, str] | None, top: int = 10) -> PlanGraph:
    """Create a tiny DAG for sum-by-group queries (e.g., 'top states by volume')."""
    nodes = [
        Step(id="f", op="filter", params={"column": "eff_gas_day", "op": "between", "value": list(date_range)}) if date_range else None,
        Step(id="a", op="aggregate", params={"groupby": group, "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]}),
        Step(id="r", op="rank", params={"by": ["sum_scheduled_quantity"], "descending": True}),
        Step(id="l", op="limit", params={"n": top}),
        Step(id="e", op="evidence_collect", params={}),
    ]
    nodes = [n for n in nodes if n is not None]
    edges = [Edge(src=nodes[i].id, dst=nodes[i+1].id) for i in range(len(nodes)-1)]
    
    # Connect raw input to first node
    if nodes:
        edges.insert(0, Edge(src="raw", dst=nodes[0].id))
    
    # Output is the last 'data' node before evidence_collect
    outputs = [nodes[-2].id] if len(nodes) > 1 and nodes[-1].op == "evidence_collect" else [nodes[-1].id] if nodes else []
    return PlanGraph(nodes=nodes, edges=edges, inputs=["raw"], outputs=outputs)


def _tiny_daily_totals(group: list[str], date_range: tuple[str, str] | None) -> PlanGraph:
    """Create a tiny DAG for daily totals queries."""
    nodes = [
        Step(id="f", op="filter", params={"column": "eff_gas_day", "op": "between", "value": list(date_range)}) if date_range else None,
        Step(id="r", op="resample", params={"freq": "1d", "on": "eff_gas_day", "agg": {"scheduled_quantity": "sum"}}),
        Step(id="a", op="aggregate", params={"groupby": group + ["eff_gas_day"], "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]}),
        Step(id="e", op="evidence_collect", params={}),
    ]
    nodes = [n for n in nodes if n is not None]
    edges = [Edge(src=nodes[i].id, dst=nodes[i+1].id) for i in range(len(nodes)-1)]
    
    # Connect raw input to first node
    if nodes:
        edges.insert(0, Edge(src="raw", dst=nodes[0].id))
    
    # Output is the last 'data' node before evidence_collect
    outputs = [nodes[-2].id] if len(nodes) > 1 and nodes[-1].op == "evidence_collect" else [nodes[-1].id] if nodes else []
    return PlanGraph(nodes=nodes, edges=edges, inputs=["raw"], outputs=outputs)


def _tiny_count_by_group(group: list[str], date_range: tuple[str, str] | None, top: int = 10) -> PlanGraph:
    """Create a tiny DAG for count-by-group queries."""
    nodes = [
        Step(id="f", op="filter", params={"column": "eff_gas_day", "op": "between", "value": list(date_range)}) if date_range else None,
        Step(id="a", op="aggregate", params={"groupby": group, "metrics": [{"col": "scheduled_quantity", "fn": "count"}]}),
        Step(id="r", op="rank", params={"by": ["count"], "descending": True}),
        Step(id="l", op="limit", params={"n": top}),
        Step(id="e", op="evidence_collect", params={}),
    ]
    nodes = [n for n in nodes if n is not None]
    edges = [Edge(src=nodes[i].id, dst=nodes[i+1].id) for i in range(len(nodes)-1)]
    
    # Connect raw input to first node
    if nodes:
        edges.insert(0, Edge(src="raw", dst=nodes[0].id))
    
    # Output is the last 'data' node before evidence_collect
    outputs = [nodes[-2].id] if len(nodes) > 1 and nodes[-1].op == "evidence_collect" else [nodes[-1].id] if nodes else []
    return PlanGraph(nodes=nodes, edges=edges, inputs=["raw"], outputs=outputs)


def build_tiny_dag_from_legacy_args(q: str, opts: dict[str, Any]) -> PlanGraph:
    """Build a tiny DAG from legacy CLI arguments.
    
    Args:
        q: Natural language question/query string
        opts: Dictionary of options parsed from CLI flags
        
    Returns:
        PlanGraph representing the compiled tiny DAG
    """
    # Parse date range from options
    date_range = opts.get("date_range")
    if not date_range and ("2022" in q or "2023" in q or "2024" in q):
        # Extract year from query and create date range
        import re
        year_match = re.search(r'\b(202[0-9])\b', q)
        if year_match:
            year = year_match.group(1)
            date_range = (f"{year}-01-01", f"{year}-12-31")
    
    # Parse top-k from options or query
    top = int(opts.get("top", 10))
    if "top" in q.lower():
        import re
        top_match = re.search(r'top\s+(\d+)', q.lower())
        if top_match:
            top = int(top_match.group(1))
    
    # Minimal intent routing based on query content
    low = q.lower()
    
    # Count-based queries (check first - highest priority)
    if any(phrase in low for phrase in ["count", "number of", "how many"]):
        if "state" in low:
            return _tiny_count_by_group(["state_abb"], date_range, top)
        elif "pipeline" in low:
            return _tiny_count_by_group(["pipeline_name"], date_range, top)
        elif "county" in low:
            return _tiny_count_by_group(["county_name"], date_range, top)
        else:
            return _tiny_count_by_group(["category_short"], date_range, top)
    
    # Daily totals queries
    if any(phrase in low for phrase in ["daily total", "daily sum", "daily volume", "per day", "by day"]):
        # Determine grouping - default to pipeline if not specified
        if "state" in low:
            group = ["state_abb"]
        elif "county" in low:
            group = ["county_name"]
        elif "category" in low:
            group = ["category_short"]
        else:
            group = ["pipeline_name"]
        return _tiny_daily_totals(group, date_range)
    
    # State-based queries
    if any(phrase in low for phrase in ["top states", "states by", "by state"]):
        return _tiny_sum_by_group(["state_abb"], date_range, top)
    
    # Pipeline-based queries  
    if any(phrase in low for phrase in ["top pipelines", "pipelines by", "by pipeline"]):
        return _tiny_sum_by_group(["pipeline_name"], date_range, top)
    
    # County-based queries
    if any(phrase in low for phrase in ["top counties", "counties by", "by county"]):
        return _tiny_sum_by_group(["county_name"], date_range, top)
    
    # Category-based queries
    if any(phrase in low for phrase in ["top categories", "categories by", "by category"]):
        return _tiny_sum_by_group(["category_short"], date_range, top)
    
    # Volume/quantity queries (default to sum by state)
    if any(phrase in low for phrase in ["volume", "quantity", "total", "sum", "gas"]):
        if "pipeline" in low:
            return _tiny_sum_by_group(["pipeline_name"], date_range, top)
        elif "county" in low:
            return _tiny_sum_by_group(["county_name"], date_range, top)
        elif "category" in low:
            return _tiny_sum_by_group(["category_short"], date_range, top)
        else:
            return _tiny_sum_by_group(["state_abb"], date_range, top)
    
    # Fallback: minimal evidence scaffold so CLI still responds
    return PlanGraph(
        nodes=[Step(id="e", op="evidence_collect", params={})], 
        edges=[Edge(src="raw", dst="e")], 
        inputs=["raw"], 
        outputs=["e"]
    )


def parse_legacy_options(args: list[str]) -> dict[str, Any]:
    """Parse legacy CLI options into a dictionary.
    
    Args:
        args: List of CLI arguments (excluding the query)
        
    Returns:
        Dictionary of parsed options
    """
    opts = {}
    
    # Simple argument parsing for common legacy flags
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg == "--since" and i + 1 < len(args):
            start_date = args[i + 1]
            opts["date_range"] = (start_date, "2030-01-01")  # Open-ended range
            i += 2
        elif arg == "--until" and i + 1 < len(args):
            end_date = args[i + 1]
            if "date_range" in opts:
                opts["date_range"] = (opts["date_range"][0], end_date)
            else:
                opts["date_range"] = ("2020-01-01", end_date)
            i += 2
        elif arg == "--top" and i + 1 < len(args):
            opts["top"] = int(args[i + 1])
            i += 2
        elif arg == "--year" and i + 1 < len(args):
            year = args[i + 1]
            opts["date_range"] = (f"{year}-01-01", f"{year}-12-31")
            i += 2
        else:
            i += 1
    
    return opts


def extract_date_range_from_query(query: str) -> tuple[str, str] | None:
    """Extract date range from natural language query.
    
    Args:
        query: Natural language query string
        
    Returns:
        Tuple of (start_date, end_date) or None if no dates found
    """
    import re
    
    # Look for date range patterns first (most specific)
    range_pattern = r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
    range_match = re.search(range_pattern, query)
    if range_match:
        return (range_match.group(1), range_match.group(2))
    
    # Look for "since" patterns (check before year pattern)
    since_pattern = r'since\s+(\d{4}-\d{2}-\d{2})'
    since_match = re.search(since_pattern, query)
    if since_match:
        start_date = since_match.group(1)
        return (start_date, "2030-01-01")
    
    # Look for year patterns last (least specific)
    year_pattern = r'\b(202[0-9])\b'
    year_match = re.search(year_pattern, query)
    if year_match:
        year = year_match.group(1)
        return (f"{year}-01-01", f"{year}-12-31")
    
    return None
