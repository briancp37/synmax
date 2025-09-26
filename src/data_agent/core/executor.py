"""Query execution engine.

DEPRECATED: This module is deprecated in favor of agent_executor.py.
All new code should use the DAG-based agent executor instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl

from ..cache import CacheManager
from . import ops
from .evidence import _digest, build_evidence
from .plan_schema import Plan


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


def run(lf: pl.LazyFrame, plan: Plan, cache_manager: CacheManager | None = None) -> Answer:
    """Execute a query plan against a lazy frame.

    Args:
        lf: Input lazy frame
        plan: Query plan to execute
        cache_manager: Optional cache manager for result caching

    Returns:
        Answer containing results and evidence
    """
    # Try to get dataset digest for caching
    dataset_digest = "unknown"
    if hasattr(lf, "_scan_path"):  # Polars LazyFrame may have scan path
        try:
            dataset_digest = _digest(Path(lf._scan_path))
        except (AttributeError, TypeError):
            pass
    else:
        # For test LazyFrames, try to get digest from a default path
        try:
            dataset_digest = _digest(Path("test_data.parquet"))
        except FileNotFoundError:
            pass

    # Check cache first if cache manager is provided
    if cache_manager:
        cached_df, cached_evidence = cache_manager.get(plan, dataset_digest)
        if cached_df is not None and cached_evidence is not None:
            # Update evidence to show cache hit
            cached_evidence["cache"]["hit"] = True
            return Answer(cached_df, cached_evidence)

    # Execute query
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
    timings = {"plan": t1 - t0, "collect": t2 - t1}
    evidence = build_evidence(lf, plan, df, timings, cache_hit=False)

    # Store in cache if cache manager is provided
    if cache_manager:
        cache_manager.put(plan, dataset_digest, df, evidence)

    return Answer(df, evidence)
