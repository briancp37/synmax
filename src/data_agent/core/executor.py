"""Query execution engine."""

from __future__ import annotations

import time
from typing import Any

import polars as pl

from .plan_schema import Plan
from . import ops
from .evidence import build_evidence


class Answer:
    """Container for query results and evidence."""
    
    def __init__(self, table: pl.DataFrame, evidence: dict[str, Any]):
        """Initialize answer with table and evidence.
        
        Args:
            table: Result dataframe
            evidence: Evidence metadata dictionary
        """
        self.table = table
        self.evidence = evidence


def run(lf: pl.LazyFrame, plan: Plan) -> Answer:
    """Execute a query plan against a lazy frame.
    
    Args:
        lf: Input lazy frame
        plan: Query plan to execute
        
    Returns:
        Answer containing results and evidence
    """
    t0 = time.perf_counter()
    out_lf = ops.apply_plan(lf, plan)
    t1 = time.perf_counter()
    
    # Collect the lazy frame
    df = out_lf.collect()
    
    # Apply sorting and limiting
    if plan.sort:
        by = plan.sort.by
        desc = plan.sort.desc
        limit = plan.sort.limit
        
        df = df.sort(by=by, descending=desc)
        if limit:
            df = df.head(limit)
    
    t2 = time.perf_counter()
    
    # Build evidence
    timings = {
        "plan": t1 - t0,
        "collect": t2 - t1
    }
    evidence = build_evidence(lf, plan, df, timings)
    
    return Answer(df, evidence)
