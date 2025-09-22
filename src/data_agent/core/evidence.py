"""Evidence card builder for query results."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from .plan_schema import Plan


def _digest(parquet_path: Path) -> str:
    """Generate a lightweight digest of a parquet file.
    
    Args:
        parquet_path: Path to parquet file
        
    Returns:
        Short hex digest of file metadata
    """
    try:
        st = parquet_path.stat()
        raw = f"{parquet_path}|{st.st_size}|{int(st.st_mtime)}".encode()
        return hashlib.sha256(raw).hexdigest()[:12]
    except FileNotFoundError:
        return "unknown"


def build_evidence(lf: pl.LazyFrame, plan: Plan, df: pl.DataFrame, timings: dict[str, float]) -> dict[str, Any]:
    """Build evidence card for query results.
    
    Args:
        lf: Original lazy frame
        plan: Query plan executed
        df: Result dataframe
        timings: Timing measurements
        
    Returns:
        Evidence dictionary
    """
    # Collect columns used in filters and aggregations
    cols_used = set()
    for f in plan.filters:
        cols_used.add(f.column)
    
    if plan.aggregate:
        for m in plan.aggregate.metrics:
            cols_used.add(m["col"])
        cols_used.update(plan.aggregate.groupby)
    
    if plan.sort:
        cols_used.update(plan.sort.by)
    
    # Compute missingness for used columns
    # For now, we'll set to None as computing null rates requires scanning the data
    missingness = {c: None for c in sorted(cols_used)}
    
    # Build evidence dictionary
    evidence = {
        "filters": [f.model_dump() for f in plan.filters],
        "aggregate": plan.aggregate.model_dump() if plan.aggregate else None,
        "sort": plan.sort.model_dump() if plan.sort else None,
        "rows_out": int(df.height),
        "columns": list(df.columns),
        "missingness": missingness,
        "timings_ms": {k: round(v * 1000, 1) for k, v in timings.items()},
        "cache": {"hit": False},
        "repro": {
            "engine": "polars",
            "snippet": _generate_repro_snippet(plan)
        },
    }
    
    return evidence


def _generate_repro_snippet(plan: Plan) -> str:
    """Generate a reproducible code snippet for the query.
    
    Args:
        plan: Query plan
        
    Returns:
        Python code snippet
    """
    lines = [
        "import polars as pl",
        "lf = pl.scan_parquet('path/to/data.parquet')",
        "res = (",
        "  lf"
    ]
    
    # Add filters
    for f in plan.filters:
        if f.op == "=":
            lines.append(f"    .filter(pl.col('{f.column}') == {repr(f.value)})")
        elif f.op == "between":
            lo, hi = f.value
            # Handle Polars date objects specially
            if hasattr(lo, 'year'):  # Polars date object
                lo_str = f"pl.date({lo.year}, {lo.month}, {lo.day})"
            else:
                lo_str = repr(lo)
            if hasattr(hi, 'year'):  # Polars date object
                hi_str = f"pl.date({hi.year}, {hi.month}, {hi.day})"
            else:
                hi_str = repr(hi)
            lines.append(f"    .filter((pl.col('{f.column}') >= {lo_str}) & (pl.col('{f.column}') <= {hi_str}))")
        elif f.op == "in":
            lines.append(f"    .filter(pl.col('{f.column}').is_in({repr(f.value)}))")
        elif f.op == "is_not_null":
            lines.append(f"    .filter(pl.col('{f.column}').is_not_null())")
        elif f.op == "contains":
            lines.append(f"    .filter(pl.col('{f.column}').cast(pl.Utf8).str.contains({repr(str(f.value))}))")
    
    # Add aggregation
    if plan.aggregate:
        if plan.aggregate.groupby:
            lines.append(f"    .group_by({repr(plan.aggregate.groupby)})")
        
        aggs = []
        for m in plan.aggregate.metrics:
            col, fn = m["col"], m["fn"].lower()
            if fn == "sum":
                aggs.append(f"pl.col('{col}').sum().alias('sum_{col}')")
            elif fn == "count":
                aggs.append("pl.len().alias('count')")
            elif fn == "avg":
                aggs.append(f"pl.col('{col}').mean().alias('avg_{col}')")
            elif fn == "p95":
                aggs.append(f"pl.col('{col}').quantile(0.95).alias('p95_{col}')")
            elif fn == "p50":
                aggs.append(f"pl.col('{col}').median().alias('p50_{col}')")
        
        if plan.aggregate.groupby:
            lines.append(f"    .agg([{', '.join(aggs)}])")
        else:
            lines.append(f"    .select([{', '.join(aggs)}])")
    
    # Add sorting
    if plan.sort:
        by = plan.sort.by
        desc = plan.sort.desc
        limit = plan.sort.limit
        
        lines.append(f"    .sort(by={repr(by)}, descending={desc})")
        if limit:
            lines.append(f"    .head({limit})")
    
    lines.extend([
        ").collect()",
        "print(res)"
    ])
    
    return "\n".join(lines)
