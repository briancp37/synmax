"""Deterministic template-based query planner with LLM support."""

import json
import os
import re
from typing import Any, Optional

from .plan_schema import Aggregate, Filter, Plan, Sort

# Pattern definitions for deterministic planning
_PATTERNS = [
    # sum deliveries for pipeline on date
    (
        re.compile(r"sum of deliveries for (.+) on (\d{4}-\d{2}-\d{2})", re.I),
        "sum_deliveries_on_date",
    ),
    # top N states by total scheduled quantity in year
    (
        re.compile(r"top (\d+) states by total scheduled quantity in (\d{4})", re.I),
        "top_states_year",
    ),
    # deliveries for pipeline in date range
    (
        re.compile(
            r"deliveries for (.+) between (\d{4}-\d{2}-\d{2}) and (\d{4}-\d{2}-\d{2})", re.I
        ),
        "deliveries_date_range",
    ),
    # total receipts for pipeline
    (re.compile(r"total receipts for (.+)", re.I), "total_receipts_pipeline"),
    # total scheduled quantity for pipeline
    (re.compile(r"total scheduled quantity for (.+?)(?:\?|$)", re.I), "total_scheduled_pipeline"),
    # average scheduled quantity by state
    (re.compile(r"average scheduled quantity by state", re.I), "avg_scheduled_by_state"),
]

# OpenAI/Anthropic function schema for Plan generation
PLAN_FUNCTION_SCHEMA = {
    "name": "generate_query_plan",
    "description": "Generate a structured query plan for gas pipeline data analysis",
    "parameters": {
        "type": "object",
        "properties": {
            "filters": {
                "type": "array",
                "description": "List of filters to apply to the data",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "description": "Column name to filter on"},
                        "op": {
                            "type": "string",
                            "enum": [
                                "=",
                                "in",
                                "between",
                                "is_not_null",
                                "contains",
                                ">=",
                                ">",
                                "<=",
                                "<",
                            ],
                            "description": "Filter operation",
                        },
                        "value": {"description": "Filter value (can be any type)"},
                    },
                    "required": ["column", "op", "value"],
                },
            },
            "resample": {
                "type": "object",
                "description": "Time-based resampling configuration",
                "properties": {
                    "freq": {"type": "string", "description": "Frequency (e.g., '1d')"},
                    "on": {"type": "string", "description": "Column to resample on"},
                },
            },
            "aggregate": {
                "type": "object",
                "description": "Aggregation configuration",
                "properties": {
                    "groupby": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to group by",
                    },
                    "metrics": {
                        "type": "array",
                        "description": "Metrics to compute",
                        "items": {
                            "type": "object",
                            "properties": {
                                "col": {"type": "string", "description": "Column name"},
                                "fn": {
                                    "type": "string",
                                    "enum": ["sum", "count", "avg", "p95", "p50"],
                                    "description": "Aggregation function",
                                },
                            },
                            "required": ["col", "fn"],
                        },
                    },
                },
            },
            "sort": {
                "type": "object",
                "description": "Sorting configuration",
                "properties": {
                    "by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to sort by",
                    },
                    "desc": {"type": "boolean", "description": "Sort in descending order"},
                    "limit": {"type": "integer", "description": "Limit number of results"},
                },
            },
            "op": {
                "type": "string",
                "enum": ["metric_compute", "changepoint", "cluster", "rules_scan", None],
                "description": "Optional advanced analytics operation",
            },
            "op_args": {"type": "object", "description": "Arguments for the analytics operation"},
            "evidence": {
                "type": "boolean",
                "description": "Whether to include evidence card",
                "default": True,
            },
            "format": {
                "type": "string",
                "enum": ["table", "json"],
                "description": "Output format",
                "default": "table",
            },
        },
        "required": [],
    },
}

# Available columns for validation
VALID_COLUMNS = [
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


def _normalize_plan_data(plan_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize plan data from LLM to match expected format."""
    import re

    import polars as pl

    # Create a copy to avoid modifying original
    normalized = plan_data.copy()

    # Normalize filters
    if "filters" in normalized:
        for filter_item in normalized["filters"]:
            column = filter_item.get("column")
            op = filter_item.get("op")
            value = filter_item.get("value")

            # Convert date strings to proper date objects for eff_gas_day
            if column == "eff_gas_day" and isinstance(value, str):
                # Handle date string formats like "2022-01-01"
                if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
                    year, month, day = map(int, value.split("-"))
                    date_val = pl.date(year, month, day)

                    if op == "=":
                        # Convert single date to between range
                        filter_item["op"] = "between"
                        filter_item["value"] = [date_val, date_val]
                    elif op in [">=", ">", "<=", "<"]:
                        # Convert comparison operators to between ranges
                        if op == ">=":
                            # >= 2023-01-01 becomes between 2023-01-01 and 2099-12-31
                            filter_item["op"] = "between"
                            filter_item["value"] = [date_val, pl.date(2099, 12, 31)]
                        elif op == ">":
                            # > 2023-01-01 becomes between 2023-01-02 and 2099-12-31
                            next_day = date_val + pl.duration(days=1)
                            filter_item["op"] = "between"
                            filter_item["value"] = [next_day, pl.date(2099, 12, 31)]
                        elif op == "<=":
                            # <= 2023-12-31 becomes between 1900-01-01 and 2023-12-31
                            filter_item["op"] = "between"
                            filter_item["value"] = [pl.date(1900, 1, 1), date_val]
                        elif op == "<":
                            # < 2024-01-01 becomes between 1900-01-01 and 2023-12-31
                            prev_day = date_val - pl.duration(days=1)
                            filter_item["op"] = "between"
                            filter_item["value"] = [pl.date(1900, 1, 1), prev_day]
                    else:
                        # For other ops, just convert the value
                        filter_item["value"] = date_val

            # Handle between with date strings
            elif (
                column == "eff_gas_day"
                and op == "between"
                and isinstance(value, list)
                and len(value) == 2
            ):
                if isinstance(value[0], str) and isinstance(value[1], str):
                    if re.match(r"^\d{4}-\d{2}-\d{2}$", value[0]) and re.match(
                        r"^\d{4}-\d{2}-\d{2}$", value[1]
                    ):
                        start_year, start_month, start_day = map(int, value[0].split("-"))
                        end_year, end_month, end_day = map(int, value[1].split("-"))
                        filter_item["value"] = [
                            pl.date(start_year, start_month, start_day),
                            pl.date(end_year, end_month, end_day),
                        ]

            # Convert string numbers to proper types for rec_del_sign
            if column == "rec_del_sign" and isinstance(value, str):
                if value == "1" or value == "+1":
                    filter_item["value"] = 1
                elif value == "-1":
                    filter_item["value"] = -1

    return normalized


def _validate_plan_json(plan_data: dict[str, Any]) -> bool:
    """Validate that the plan JSON contains valid column references and operations."""
    try:
        # Check filters reference valid columns
        if "filters" in plan_data:
            for filter_item in plan_data["filters"]:
                if filter_item.get("column") not in VALID_COLUMNS:
                    return False
                if filter_item.get("op") not in [
                    "=",
                    "in",
                    "between",
                    "is_not_null",
                    "contains",
                    ">=",
                    ">",
                    "<=",
                    "<",
                ]:
                    return False

        # Check aggregate references valid columns
        if "aggregate" in plan_data and plan_data["aggregate"]:
            agg = plan_data["aggregate"]
            if "groupby" in agg:
                for col in agg["groupby"]:
                    if col not in VALID_COLUMNS:
                        return False
            if "metrics" in agg:
                for metric in agg["metrics"]:
                    if metric.get("col") not in VALID_COLUMNS:
                        return False
                    if metric.get("fn") not in ["sum", "count", "avg", "p95", "p50"]:
                        return False

        # Check sort references valid columns
        if "sort" in plan_data and plan_data["sort"]:
            sort_data = plan_data["sort"]
            if "by" in sort_data:
                for col in sort_data["by"]:
                    # Allow computed column names like sum_scheduled_quantity
                    if not (col in VALID_COLUMNS or "_" in col):
                        return False

        return True
    except (KeyError, TypeError, AttributeError):
        return False


def _call_openai_planner(query: str) -> Optional[dict[str, Any]]:
    """Call OpenAI API to generate a plan."""
    try:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(  # type: ignore
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a data analyst specializing in gas pipeline data. 
Generate structured query plans for natural language questions about pipeline data.

Available columns:
- pipeline_name: Name of the pipeline
- loc_name: Location name  
- connecting_pipeline: Connected pipeline name
- connecting_entity: Connected entity name
- rec_del_sign: Receipt (+1) or delivery (-1) indicator
- category_short: Category (e.g., Interconnect, LDC, Industrial)
- country_name: Country name
- state_abb: Two-letter state abbreviation
- county_name: County name
- latitude, longitude: Geographic coordinates
- eff_gas_day: Effective gas day (date)
- scheduled_quantity: Daily scheduled flow quantity

Use the generate_query_plan function to create a structured plan.""",
                },
                {"role": "user", "content": f"Generate a query plan for: {query}"},
            ],
            tools=[{"type": "function", "function": PLAN_FUNCTION_SCHEMA}],
            tool_choice={"type": "function", "function": {"name": "generate_query_plan"}},
        )

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function:
                return json.loads(tool_call.function.arguments)  # type: ignore
    except Exception:
        pass
    return None


def _call_anthropic_planner(query: str) -> Optional[dict[str, Any]]:
    """Call Anthropic API to generate a plan."""
    try:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        client = anthropic.Anthropic(api_key=api_key)

        # Convert OpenAI schema to Anthropic tool format
        tool = {
            "name": "generate_query_plan",
            "description": "Generate a structured query plan for gas pipeline data analysis",
            "input_schema": PLAN_FUNCTION_SCHEMA["parameters"],
        }

        message = client.messages.create(  # type: ignore
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a data analyst specializing in gas pipeline data. 
Generate structured query plans for natural language questions about pipeline data.

Available columns:
- pipeline_name: Name of the pipeline
- loc_name: Location name  
- connecting_pipeline: Connected pipeline name
- connecting_entity: Connected entity name
- rec_del_sign: Receipt (+1) or delivery (-1) indicator
- category_short: Category (e.g., Interconnect, LDC, Industrial)
- country_name: Country name
- state_abb: Two-letter state abbreviation
- county_name: County name
- latitude, longitude: Geographic coordinates
- eff_gas_day: Effective gas day (date)
- scheduled_quantity: Daily scheduled flow quantity

Generate a query plan for: {query}""",
                }
            ],
            model="claude-3-sonnet-20240229",
            tools=[tool],
            tool_choice={"type": "tool", "name": "generate_query_plan"},
        )

        if message.content and message.content[0].type == "tool_use":
            return message.content[0].input  # type: ignore
    except Exception:
        pass
    return None


def _llm_plan(query: str) -> Optional[Plan]:
    """Generate a plan using LLM (OpenAI or Anthropic)."""
    # Try OpenAI first
    plan_data = _call_openai_planner(query)

    # Validate and normalize OpenAI response
    if plan_data is not None and _validate_plan_json(plan_data):
        try:
            normalized_data = _normalize_plan_data(plan_data)
            return Plan(**normalized_data)
        except Exception:
            pass  # Fall through to try Anthropic

    # If OpenAI fails or returns invalid data, try Anthropic
    plan_data = _call_anthropic_planner(query)

    # Validate and normalize Anthropic response
    if plan_data is not None and _validate_plan_json(plan_data):
        try:
            normalized_data = _normalize_plan_data(plan_data)
            return Plan(**normalized_data)
        except Exception:
            pass

    # Both failed
    return None


def plan(q: str, deterministic: bool = True) -> Plan:
    """Create a query plan from natural language input.

    Args:
        q: Natural language query string
        deterministic: If True, use template-based planning; if False, use LLM planning

    Returns:
        Plan object representing the query execution plan

    Raises:
        ValueError: If no matching pattern is found for deterministic planning
    """
    if not deterministic:
        # Try LLM-based planning first
        llm_result = _llm_plan(q)
        if llm_result is not None:
            return llm_result
        # Fallback to deterministic if LLM fails
        # Continue with deterministic planning below

    for rx, key in _PATTERNS:
        m = rx.search(q)
        if not m:
            continue

        if key == "sum_deliveries_on_date":
            pipeline, d = m.group(1).strip(), m.group(2)
            import polars as pl

            year, month, day = map(int, d.split("-"))
            date_val = pl.date(year, month, day)
            return Plan(
                filters=[
                    Filter(column="pipeline_name", op="=", value=pipeline),
                    Filter(column="rec_del_sign", op="=", value=-1),  # -1 for deliveries
                    Filter(column="eff_gas_day", op="between", value=[date_val, date_val]),
                ],
                aggregate=Aggregate(
                    groupby=[], metrics=[{"col": "scheduled_quantity", "fn": "sum"}]
                ),
            )

        elif key == "top_states_year":
            n = int(m.group(1))
            year_str = m.group(2)
            year = int(year_str)
            import polars as pl

            return Plan(
                filters=[
                    Filter(
                        column="eff_gas_day",
                        op="between",
                        value=[pl.date(year, 1, 1), pl.date(year, 12, 31)],
                    )
                ],
                aggregate=Aggregate(
                    groupby=["state_abb"], metrics=[{"col": "scheduled_quantity", "fn": "sum"}]
                ),
                sort=Sort(by=["sum_scheduled_quantity"], desc=True, limit=n),
            )

        elif key == "deliveries_date_range":
            pipeline, start_date, end_date = m.group(1).strip(), m.group(2), m.group(3)
            import polars as pl

            start_year, start_month, start_day = map(int, start_date.split("-"))
            end_year, end_month, end_day = map(int, end_date.split("-"))
            start_date_val = pl.date(start_year, start_month, start_day)
            end_date_val = pl.date(end_year, end_month, end_day)
            return Plan(
                filters=[
                    Filter(column="pipeline_name", op="=", value=pipeline),
                    Filter(column="rec_del_sign", op="=", value=-1),  # -1 for deliveries
                    Filter(
                        column="eff_gas_day", op="between", value=[start_date_val, end_date_val]
                    ),
                ],
                aggregate=Aggregate(
                    groupby=["eff_gas_day"], metrics=[{"col": "scheduled_quantity", "fn": "sum"}]
                ),
                sort=Sort(by=["eff_gas_day"], desc=False),
            )

        elif key == "total_receipts_pipeline":
            pipeline = m.group(1).strip()
            return Plan(
                filters=[
                    Filter(column="pipeline_name", op="=", value=pipeline),
                    Filter(column="rec_del_sign", op="=", value=1),  # 1 for receipts
                ],
                aggregate=Aggregate(
                    groupby=[], metrics=[{"col": "scheduled_quantity", "fn": "sum"}]
                ),
            )

        elif key == "total_scheduled_pipeline":
            pipeline = m.group(1).strip()
            return Plan(
                filters=[
                    Filter(column="pipeline_name", op="=", value=pipeline),
                ],
                aggregate=Aggregate(
                    groupby=[], metrics=[{"col": "scheduled_quantity", "fn": "sum"}]
                ),
            )

        elif key == "avg_scheduled_by_state":
            return Plan(
                aggregate=Aggregate(
                    groupby=["state_abb"], metrics=[{"col": "scheduled_quantity", "fn": "avg"}]
                ),
                sort=Sort(by=["avg_scheduled_quantity"], desc=True),
            )

    # Fallback: return minimal plan if no pattern matches
    return Plan()
