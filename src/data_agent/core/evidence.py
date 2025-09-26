"""Evidence card builder for query results."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ..config import ARTIFACTS_DIR
from ..rules.engine import run_rules
from .agent_schema import PlanGraph
from .plan_schema import Plan


@dataclass
class StepEvidence:
    """Evidence for a single step execution."""

    node_id: str
    params: dict[str, Any]
    input_stats: dict[str, Any]  # rows, bytes, columns from input handles
    output_stats: dict[str, Any]  # rows, bytes, columns from output
    timings: dict[str, float]  # execution timings in seconds
    snippet: str  # reproducible code snippet
    checkpoint_path: str | None = None  # path to checkpoint file if materialized


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


def build_evidence(
    lf: pl.LazyFrame,
    plan: Plan,
    df: pl.DataFrame,
    timings: dict[str, float],
    cache_hit: bool = False,
) -> dict[str, Any]:
    """Build evidence card for query results.

    Args:
        lf: Original lazy frame
        plan: Query plan executed
        df: Result dataframe
        timings: Timing measurements
        cache_hit: Whether this result was served from cache

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

    # Run data quality rules and include summary
    rules_results = run_rules(lf)
    rules_summary = {
        rule_id: {"count": result["count"], "samples": result["samples"][:3]}
        for rule_id, result in rules_results.items()
        if result["count"] > 0
    }

    # Build evidence dictionary with serializable filter values
    def _serialize_filter_value(value: Any) -> Any:
        """Convert filter values to serializable format."""
        if hasattr(value, "__class__") and "Expr" in str(value.__class__):
            return str(value)
        elif isinstance(value, (list, tuple)):
            return [_serialize_filter_value(v) for v in value]
        else:
            return value

    serializable_filters = []
    for f in plan.filters:
        filter_dict = f.model_dump()
        filter_dict["value"] = _serialize_filter_value(filter_dict["value"])
        serializable_filters.append(filter_dict)

    evidence = {
        "filters": serializable_filters,
        "aggregate": plan.aggregate.model_dump() if plan.aggregate else None,
        "sort": plan.sort.model_dump() if plan.sort else None,
        "operation": {"type": plan.op, "parameters": plan.op_args} if plan.op else None,
        "rows_out": int(df.height),
        "columns": list(df.columns),
        "missingness": missingness,
        "rules": rules_summary,
        "timings_ms": {k: round(v * 1000, 1) for k, v in timings.items()},
        "cache": {"hit": cache_hit},
        "repro": {"engine": "polars", "snippet": _generate_repro_snippet(plan)},
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
        "  lf",
    ]

    # Add filters
    for f in plan.filters:
        if f.op == "=":
            lines.append(f"    .filter(pl.col('{f.column}') == {repr(f.value)})")
        elif f.op == "between":
            lo, hi = f.value
            # Handle Polars date objects specially
            if hasattr(lo, "year"):  # Polars date object
                lo_str = f"pl.date({lo.year}, {lo.month}, {lo.day})"
            else:
                lo_str = repr(lo)
            if hasattr(hi, "year"):  # Polars date object
                hi_str = f"pl.date({hi.year}, {hi.month}, {hi.day})"
            else:
                hi_str = repr(hi)
            filter_expr = f"(pl.col('{f.column}') >= {lo_str}) & (pl.col('{f.column}') <= {hi_str})"
            lines.append(f"    .filter({filter_expr})")
        elif f.op == "in":
            lines.append(f"    .filter(pl.col('{f.column}').is_in({repr(f.value)}))")
        elif f.op == "is_not_null":
            lines.append(f"    .filter(pl.col('{f.column}').is_not_null())")
        elif f.op == "contains":
            contains_expr = f"pl.col('{f.column}').cast(pl.Utf8).str.contains({repr(str(f.value))})"
            lines.append(f"    .filter({contains_expr})")

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

    lines.extend([").collect()", "print(res)"])

    return "\n".join(lines)


def save_plan_evidence(
    plan: PlanGraph,
    step_evidence: list[StepEvidence],
    final_result: pl.DataFrame,
    total_time: float,
    output_path: Path | None = None,
) -> Path:
    """Save comprehensive evidence including plan JSON and per-step evidence.

    Args:
        plan: The executed plan graph
        step_evidence: List of per-step evidence
        final_result: Final result DataFrame
        total_time: Total execution time in seconds
        output_path: Optional output path (defaults to artifacts/outputs/<plan_hash>.json)

    Returns:
        Path where evidence was saved
    """
    plan_hash = plan.plan_hash()

    if output_path is None:
        outputs_dir = ARTIFACTS_DIR / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outputs_dir / f"{plan_hash}.json"

    # Convert step evidence to serializable format
    step_evidence_dicts = []
    for evidence in step_evidence:
        step_evidence_dicts.append(
            {
                "node_id": evidence.node_id,
                "params": evidence.params,
                "input_stats": evidence.input_stats,
                "output_stats": evidence.output_stats,
                "timings": evidence.timings,
                "snippet": evidence.snippet,
                "checkpoint_path": evidence.checkpoint_path,
            }
        )

    # Build comprehensive evidence document
    evidence_doc = {
        "plan": {
            "plan_hash": plan_hash,
            "nodes": [
                {
                    "id": step.id,
                    "op": step.op.value,
                    "params": step.params,
                    "engine": step.engine,
                    "materialize": step.materialize,
                }
                for step in plan.nodes
            ],
            "edges": [{"src": edge.src, "dst": edge.dst} for edge in plan.edges],
            "inputs": plan.inputs,
            "outputs": plan.outputs,
        },
        "steps": step_evidence_dicts,
        "final_result": {
            "rows": final_result.height,
            "columns": final_result.width,
            "column_names": final_result.columns,
        },
        "metadata": {
            "total_steps": len(step_evidence),
            "total_time_seconds": total_time,
            "created_at": time.time(),
            "replay_token": plan_hash,
        },
    }

    # Write evidence file
    with open(output_path, "w") as f:
        json.dump(evidence_doc, f, indent=2, default=str)

    return output_path


def generate_step_code_snippet(
    step_type: str, params: dict[str, Any], input_path: str | None = None
) -> str:
    """Generate reproducible code snippet for a specific step type.

    Args:
        step_type: The type of step (filter, aggregate, etc.)
        params: Step parameters
        input_path: Path to input data file

    Returns:
        Python code snippet for reproducing this step
    """
    lines = [
        "import polars as pl",
        f"# Step: {step_type}",
    ]

    # Add data loading
    if input_path:
        lines.append(f'lf = pl.scan_parquet("{input_path}")')
    else:
        lines.append("lf = pl.scan_parquet('input.parquet')")

    # Generate step-specific code
    if step_type == "filter":
        column = params.get("column", "column")
        op = params.get("op", "=")
        value = params.get("value")

        if op == "=":
            lines.append(f"result = lf.filter(pl.col('{column}') == {repr(value)})")
        elif op == "!=":
            lines.append(f"result = lf.filter(pl.col('{column}') != {repr(value)})")
        elif op == "in":
            lines.append(f"result = lf.filter(pl.col('{column}').is_in({repr(value)}))")
        elif op == "between":
            if isinstance(value, list) and len(value) == 2:
                lo, hi = value
                lines.append(
                    f"result = lf.filter((pl.col('{column}') >= {repr(lo)}) & "
                    f"(pl.col('{column}') <= {repr(hi)}))"
                )
        elif op == "is_not_null":
            lines.append(f"result = lf.filter(pl.col('{column}').is_not_null())")
        elif op == "contains":
            filter_expr = f"pl.col('{column}').cast(pl.Utf8).str.contains({repr(str(value))})"
            lines.append(f"result = lf.filter({filter_expr})")
        elif op in [">", "<", ">=", "<="]:
            lines.append(f"result = lf.filter(pl.col('{column}') {op} {repr(value)})")
        else:
            lines.append(f"# Filter operation: {op} with value: {value}")
            lines.append("result = lf")

    elif step_type == "aggregate":
        groupby = params.get("groupby", [])
        metrics = params.get("metrics", [])

        agg_exprs = []
        for metric in metrics:
            col = metric.get("col", "column")
            fn = metric.get("fn", "sum").lower()

            if fn == "sum":
                agg_exprs.append(f"pl.col('{col}').sum().alias('sum_{col}')")
            elif fn == "count":
                agg_exprs.append("pl.len().alias('count')")
            elif fn == "avg":
                agg_exprs.append(f"pl.col('{col}').mean().alias('avg_{col}')")
            elif fn == "p95":
                agg_exprs.append(f"pl.col('{col}').quantile(0.95).alias('p95_{col}')")
            elif fn == "p50":
                agg_exprs.append(f"pl.col('{col}').median().alias('p50_{col}')")
            elif fn in ["min", "max", "std"]:
                agg_exprs.append(f"pl.col('{col}').{fn}().alias('{fn}_{col}')")
            else:
                agg_exprs.append(f"# Unknown function: {fn}")

        if groupby:
            lines.append(f"result = lf.group_by({repr(groupby)}).agg([{', '.join(agg_exprs)}])")
        else:
            lines.append(f"result = lf.select([{', '.join(agg_exprs)}])")

    elif step_type == "resample":
        freq = params.get("freq", "1d")
        on = params.get("on", "timestamp")
        agg = params.get("agg", {})

        if agg:
            agg_exprs = []
            for col, fn in agg.items():
                if fn == "sum":
                    agg_exprs.append(f"pl.col('{col}').sum()")
                elif fn == "mean":
                    agg_exprs.append(f"pl.col('{col}').mean()")
                elif fn == "count":
                    agg_exprs.append("pl.len()")
                else:
                    agg_exprs.append(f"pl.col('{col}').first()")
            agg_expr = f"[{', '.join(agg_exprs)}]"
            lines.append(f"result = lf.group_by_dynamic('{on}', every='{freq}').agg({agg_expr})")
        else:
            lines.append(
                f"result = lf.group_by_dynamic('{on}', every='{freq}').agg(pl.all().first())"
            )

    elif step_type == "stl_deseasonalize":
        column = params.get("column", "value")
        period = params.get("period", 7)
        seasonal = params.get("seasonal", None)
        trend = params.get("trend", None)

        lines.append(f"# STL Deseasonalization on column '{column}'")
        lines.append(f"# Period: {period}, Seasonal: {seasonal}, Trend: {trend}")
        lines.append("# Note: This requires statsmodels.tsa.seasonal.STL")
        lines.append("result = lf  # Placeholder - actual STL implementation needed")

    elif step_type == "changepoint":
        column = params.get("column", "value")
        method = params.get("method", "pelt")
        min_size = params.get("min_size", 2)

        lines.append(f"# Changepoint detection on column '{column}'")
        lines.append(f"# Method: {method}, Min size: {min_size}")
        lines.append("# Note: This requires ruptures library")
        lines.append("result = lf  # Placeholder - actual changepoint implementation needed")

    elif step_type == "rank":
        by = params.get("by", ["value"])
        method = params.get("method", "average")
        descending = params.get("descending", False)

        rank_exprs = []
        for col in by:
            rank_expr = (
                f"pl.col('{col}').rank(method='{method}', descending={descending})"
                f".alias('rank_{col}')"
            )
            rank_exprs.append(rank_expr)

        lines.append(f"result = lf.with_columns([{', '.join(rank_exprs)}])")

    elif step_type == "limit":
        n = params.get("n", 10)
        offset = params.get("offset", 0)

        if offset > 0:
            lines.append(f"result = lf.slice({offset}, {n})")
        else:
            lines.append(f"result = lf.head({n})")

    elif step_type == "save_artifact":
        path = params.get("path", "output.parquet")
        format_type = params.get("format", "parquet")

        lines.append(f"# Save to {path} as {format_type}")
        if format_type == "parquet":
            lines.append(f"lf.collect().write_parquet('{path}')")
        elif format_type == "csv":
            lines.append(f"lf.collect().write_csv('{path}')")
        elif format_type == "json":
            lines.append(f"lf.collect().write_json('{path}')")
        lines.append("result = lf")

    elif step_type == "evidence_collect":
        sample_size = params.get("sample_size", 100)
        method = params.get("method", "random")

        lines.append(f"# Evidence collection - sample {sample_size} rows using {method}")
        if method == "random":
            lines.append(f"result = lf.collect().sample(n={sample_size}).lazy()")
        elif method == "head":
            lines.append(f"result = lf.head({sample_size})")
        elif method == "tail":
            lines.append(f"result = lf.tail({sample_size})")
        else:
            lines.append("result = lf")

    else:
        lines.append(f"# Unknown step type: {step_type}")
        lines.append("result = lf")

    lines.append("print(result.collect())")
    return "\n".join(lines)
