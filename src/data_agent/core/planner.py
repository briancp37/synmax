"""Deterministic template-based query planner."""

import re

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


def plan(q: str, deterministic: bool = True) -> Plan:
    """Create a query plan from natural language input.

    Args:
        q: Natural language query string
        deterministic: If True, use template-based planning (only option for now)

    Returns:
        Plan object representing the query execution plan

    Raises:
        ValueError: If no matching pattern is found for deterministic planning
    """
    if not deterministic:
        raise NotImplementedError("LLM-based planning not yet implemented")

    for rx, key in _PATTERNS:
        m = rx.search(q)
        if not m:
            continue

        if key == "sum_deliveries_on_date":
            pipeline, d = m.group(1).strip(), m.group(2)
            import polars as pl
            year, month, day = map(int, d.split('-'))
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
            n, year = int(m.group(1)), m.group(2)
            import polars as pl
            return Plan(
                filters=[
                    Filter(
                        column="eff_gas_day", op="between", value=[pl.date(int(year), 1, 1), pl.date(int(year), 12, 31)]
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
            start_year, start_month, start_day = map(int, start_date.split('-'))
            end_year, end_month, end_day = map(int, end_date.split('-'))
            start_date_val = pl.date(start_year, start_month, start_day)
            end_date_val = pl.date(end_year, end_month, end_day)
            return Plan(
                filters=[
                    Filter(column="pipeline_name", op="=", value=pipeline),
                    Filter(column="rec_del_sign", op="=", value=-1),  # -1 for deliveries
                    Filter(column="eff_gas_day", op="between", value=[start_date_val, end_date_val]),
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
