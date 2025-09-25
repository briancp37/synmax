"""Agent executor for DAG-based query plans."""

from __future__ import annotations

import time
from typing import Any

import polars as pl

from ..config import ARTIFACTS_DIR, BYTE_CKPT, ROW_CKPT
from .agent_schema import PlanGraph, StepType
from .evidence import StepEvidence
from .handles import HandleStorage, StepHandle, StepStats, create_lazy_handle


class AgentExecutor:
    """Executes DAG plans with step-by-step evidence collection."""

    def __init__(self, handle_storage: HandleStorage | None = None):
        """Initialize the executor.

        Args:
            handle_storage: Optional handle storage manager
        """
        self.handle_storage = handle_storage or HandleStorage()
        self.step_evidence: list[StepEvidence] = []
        self.lazy_frames: dict[str, pl.LazyFrame] = {}  # Store lazy frames by handle ID

    def execute(
        self, plan: PlanGraph, dataset_handle: StepHandle
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Execute a DAG plan and return results with evidence.

        Args:
            plan: DAG plan to execute
            dataset_handle: Handle to input dataset

        Returns:
            Tuple of (final_table, evidence_dict)
        """
        # Reset evidence collection
        self.step_evidence = []

        # Get topological order for execution
        topo_order = plan.topological_order()

        # Track handles for each node
        node_handles: dict[str, StepHandle] = {}

        # Add input handles and lazy frames
        for input_id in plan.inputs:
            if input_id == "raw":
                node_handles[input_id] = dataset_handle
                # Initialize lazy frame for raw data
                if dataset_handle.store == "parquet" and dataset_handle.path:
                    self.lazy_frames[input_id] = pl.scan_parquet(dataset_handle.path)
                else:
                    raise ValueError(f"Cannot initialize lazy frame for handle: {dataset_handle}")
            else:
                raise ValueError(f"Unknown input source: {input_id}")

        # Execute each step in topological order
        for node_id in topo_order:
            # Skip input nodes - they're already handled
            if node_id in plan.inputs:
                continue

            # Find the step definition
            step = next((s for s in plan.nodes if s.id == node_id), None)
            if not step:
                raise ValueError(f"Step not found: {node_id}")

            # Find input handles for this step
            input_handles = []
            for edge in plan.edges:
                if edge.dst == node_id:
                    if edge.src not in node_handles:
                        raise ValueError(f"Input handle not found: {edge.src}")
                    input_handles.append(node_handles[edge.src])

            if not input_handles:
                raise ValueError(f"No inputs found for step: {node_id}")

            # Execute the step
            output_handle = self._execute_step(step, input_handles, plan)
            node_handles[node_id] = output_handle

        # Get final outputs
        if not plan.outputs:
            # If no outputs specified, use the last step in topological order
            final_node_id = topo_order[-1]
        else:
            final_node_id = plan.outputs[0]  # Assuming single output for now

        # Collect final result
        if final_node_id in self.lazy_frames:
            final_df = self.lazy_frames[final_node_id].collect()
        elif final_node_id in node_handles:
            final_handle = node_handles[final_node_id]
            if final_handle.store == "parquet" and final_handle.path:
                final_df = pl.read_parquet(final_handle.path)
            else:
                final_df = pl.DataFrame()
        else:
            final_df = pl.DataFrame()

        # Build comprehensive evidence
        evidence = self._build_final_evidence(plan, final_df)

        return final_df, evidence

    def _execute_step(
        self, step: Any, input_handles: list[StepHandle], plan: PlanGraph
    ) -> StepHandle:
        """Execute a single step.

        Args:
            step: Step definition from the plan
            input_handles: Input handles for this step
            plan: Full plan context

        Returns:
            Output handle for this step
        """
        t0 = time.perf_counter()

        # Collect input statistics
        input_stats = self._collect_input_stats(input_handles)

        # Load input data (for now, assume single input)
        if len(input_handles) != 1:
            raise NotImplementedError("Multi-input steps not yet supported")

        input_handle = input_handles[0]

        # Get the lazy frame for the input
        input_id = input_handles[0].id
        if input_id in self.lazy_frames:
            lf = self.lazy_frames[input_id]
        elif input_handle.store == "parquet" and input_handle.path:
            lf = pl.scan_parquet(input_handle.path)
        else:
            raise ValueError(f"Cannot load data for handle: {input_handle}")

        t1 = time.perf_counter()

        # Execute the operation based on step type
        output_lf = self._execute_operation(step, lf)

        t2 = time.perf_counter()

        # Collect to get output statistics
        output_df = output_lf.collect()
        output_stats = self._collect_output_stats(output_df)

        t3 = time.perf_counter()

        # Determine if we should checkpoint this step
        consumer_count = sum(1 for edge in plan.edges if edge.src == step.id)
        should_checkpoint = self.handle_storage.should_checkpoint(
            StepStats(
                rows=output_df.height,
                bytes=int(output_df.estimated_size("mb") * 1024 * 1024),
                columns=output_df.width,
                null_count={col: output_df[col].null_count() for col in output_df.columns},
                computed_at=time.time(),
            ),
            step.op.value,
            consumer_count > 1,
            ROW_CKPT,
            BYTE_CKPT,
        )

        # Create output handle
        if should_checkpoint or step.materialize:
            # Compute fingerprint for content addressing
            fingerprint = self.handle_storage.compute_fingerprint(
                step.id, step.params, [h.fingerprint or "unknown" for h in input_handles], "dataset"
            )

            # Materialize to disk
            output_handle = self.handle_storage.materialize_handle(
                output_df, fingerprint, step.id, step.engine
            )
        else:
            # Keep as lazy handle and store the lazy frame
            output_handle = create_lazy_handle(step.id, engine=step.engine)
            self.lazy_frames[step.id] = output_lf

        t4 = time.perf_counter()

        # Generate code snippet
        snippet = self._generate_step_snippet(step, input_handle)

        # Collect step evidence
        step_evidence = StepEvidence(
            node_id=step.id,
            params=step.params,
            input_stats=input_stats,
            output_stats=output_stats,
            timings={
                "load": t1 - t0,
                "execute": t2 - t1,
                "collect": t3 - t2,
                "materialize": t4 - t3,
                "total": t4 - t0,
            },
            snippet=snippet,
            checkpoint_path=str(output_handle.path) if output_handle.path else None,
        )

        self.step_evidence.append(step_evidence)

        return output_handle

    def _execute_operation(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute a single operation on a lazy frame.

        Args:
            step: Step definition
            lf: Input lazy frame

        Returns:
            Transformed lazy frame
        """
        if step.op == StepType.FILTER:
            return self._execute_filter(step, lf)
        elif step.op == StepType.AGGREGATE:
            return self._execute_aggregate(step, lf)
        elif step.op == StepType.RESAMPLE:
            return self._execute_resample(step, lf)
        elif step.op == StepType.LIMIT:
            return self._execute_limit(step, lf)
        elif step.op == StepType.RANK:
            return self._execute_rank(step, lf)
        elif step.op == StepType.EVIDENCE_COLLECT:
            return self._execute_evidence_collect(step, lf)
        elif step.op == StepType.STL_DESEASONALIZE:
            return self._execute_stl_deseasonalize(step, lf)
        elif step.op == StepType.CHANGEPOINT:
            return self._execute_changepoint(step, lf)
        elif step.op == StepType.SAVE_ARTIFACT:
            return self._execute_save_artifact(step, lf)
        else:
            raise NotImplementedError(f"Step type not implemented: {step.op}")

    def _execute_filter(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute filter operation."""
        params = step.params
        column = params["column"]
        op = params["op"]
        value = params.get("value")

        if op == "=":
            return lf.filter(pl.col(column) == value)
        elif op == "!=":
            return lf.filter(pl.col(column) != value)
        elif op == "in":
            return lf.filter(pl.col(column).is_in(value))
        elif op == "between":
            lo, hi = value
            return lf.filter((pl.col(column) >= lo) & (pl.col(column) <= hi))
        elif op == "is_not_null":
            return lf.filter(pl.col(column).is_not_null())
        elif op == "contains":
            return lf.filter(pl.col(column).cast(pl.Utf8).str.contains(str(value)))
        elif op == ">":
            return lf.filter(pl.col(column) > value)
        elif op == "<":
            return lf.filter(pl.col(column) < value)
        elif op == ">=":
            return lf.filter(pl.col(column) >= value)
        elif op == "<=":
            return lf.filter(pl.col(column) <= value)
        else:
            raise ValueError(f"Unknown filter operation: {op}")

    def _execute_aggregate(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute aggregate operation."""
        params = step.params
        groupby = params.get("groupby", [])
        metrics = params.get("metrics", [])

        aggs = []
        for metric in metrics:
            col = metric["col"]
            fn = metric["fn"].lower()

            if fn == "sum":
                aggs.append(pl.col(col).sum().alias(f"sum_{col}"))
            elif fn == "count":
                aggs.append(pl.len().alias("count"))
            elif fn == "avg":
                aggs.append(pl.col(col).mean().alias(f"avg_{col}"))
            elif fn == "p95":
                aggs.append(pl.col(col).quantile(0.95).alias(f"p95_{col}"))
            elif fn == "p50":
                aggs.append(pl.col(col).median().alias(f"p50_{col}"))
            elif fn == "min":
                aggs.append(pl.col(col).min().alias(f"min_{col}"))
            elif fn == "max":
                aggs.append(pl.col(col).max().alias(f"max_{col}"))
            elif fn == "std":
                aggs.append(pl.col(col).std().alias(f"std_{col}"))
            else:
                raise ValueError(f"Unknown aggregation function: {fn}")

        if groupby:
            return lf.group_by(groupby).agg(aggs)
        else:
            return lf.select(aggs)

    def _execute_resample(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute resample operation."""
        params = step.params
        freq = params["freq"]
        on = params["on"]
        agg = params.get("agg", {})

        # For now, implement basic daily resampling
        if freq == "1d":
            if agg:
                agg_exprs = []
                for col, fn in agg.items():
                    if fn == "sum":
                        agg_exprs.append(pl.col(col).sum())
                    elif fn == "mean":
                        agg_exprs.append(pl.col(col).mean())
                    elif fn == "count":
                        agg_exprs.append(pl.len())
                    else:
                        agg_exprs.append(pl.col(col).first())

                return lf.group_by_dynamic(on, every=freq).agg(agg_exprs)
            else:
                return lf.group_by_dynamic(on, every=freq).agg(pl.all().first())
        else:
            raise NotImplementedError(f"Frequency not supported: {freq}")

    def _execute_limit(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute limit operation."""
        params = step.params
        n = params["n"]
        offset = params.get("offset", 0)

        if offset > 0:
            return lf.slice(offset, n)
        else:
            return lf.head(n)

    def _execute_rank(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute rank operation."""
        params = step.params
        by = params["by"]
        method = params.get("method", "average")
        descending = params.get("descending", False)

        # Add rank columns
        rank_exprs = []
        for col in by:
            rank_expr = pl.col(col).rank(method=method, descending=descending).alias(f"rank_{col}")
            rank_exprs.append(rank_expr)

        return lf.with_columns(rank_exprs)

    def _execute_evidence_collect(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute evidence collection (pass-through with sampling)."""
        params = step.params
        sample_size = params.get("sample_size", 100)
        method = params.get("method", "random")

        # For evidence collection, we might want to sample the data
        if method == "random":
            # LazyFrame doesn't have sample, so we collect first then sample
            df = lf.collect()
            if df.height > sample_size:
                return df.sample(n=sample_size).lazy()
            else:
                return df.lazy()
        elif method == "head":
            return lf.head(sample_size)
        elif method == "tail":
            return lf.tail(sample_size)
        else:
            return lf  # Pass through

    def _execute_stl_deseasonalize(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute STL deseasonalization (placeholder)."""
        # This would require implementing STL decomposition
        # For now, return the original data with a placeholder column
        return lf.with_columns(pl.col(step.params["column"]).alias("deseasonalized"))

    def _execute_changepoint(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute changepoint detection (placeholder)."""
        # This would require implementing changepoint detection algorithms
        # For now, return a placeholder result
        return lf.with_columns(pl.lit(False).alias("is_changepoint"))

    def _execute_save_artifact(self, step: Any, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Execute save artifact operation."""
        params = step.params
        path = params["path"]
        format_type = params.get("format", "parquet")
        # overwrite = params.get("overwrite", True)  # TODO: Implement overwrite logic

        # Collect and save
        df = lf.collect()
        save_path = ARTIFACTS_DIR / path

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "parquet":
            df.write_parquet(save_path)
        elif format_type == "csv":
            df.write_csv(save_path)
        elif format_type == "json":
            df.write_json(save_path)
        else:
            raise ValueError(f"Unknown format: {format_type}")

        return lf  # Pass through

    def _collect_input_stats(self, input_handles: list[StepHandle]) -> dict[str, Any]:
        """Collect statistics from input handles."""
        total_rows = 0
        total_bytes = 0
        total_columns = 0

        for handle in input_handles:
            if handle.stats:
                total_rows += handle.stats.rows
                total_bytes += handle.stats.bytes
                total_columns += handle.stats.columns

        return {
            "handles": len(input_handles),
            "total_rows": total_rows,
            "total_bytes": total_bytes,
            "total_columns": total_columns,
        }

    def _collect_output_stats(self, df: pl.DataFrame) -> dict[str, Any]:
        """Collect statistics from output DataFrame."""
        return {
            "rows": df.height,
            "columns": df.width,
            "bytes": int(df.estimated_size("mb") * 1024 * 1024),
            "null_counts": {col: df[col].null_count() for col in df.columns},
        }

    def _generate_step_snippet(self, step: Any, input_handle: StepHandle) -> str:
        """Generate reproducible code snippet for a step."""
        lines = [
            "import polars as pl",
            f"# Step: {step.id} ({step.op})",
        ]

        # Add data loading
        if input_handle.path:
            lines.append(f'lf = pl.scan_parquet("{input_handle.path}")')
        else:
            lines.append("lf = pl.scan_parquet('input.parquet')")

        # Add operation-specific code
        if step.op == StepType.FILTER:
            params = step.params
            column = params["column"]
            op = params["op"]
            value = params.get("value")

            if op == "=":
                lines.append(f"result = lf.filter(pl.col('{column}') == {repr(value)})")
            elif op == "between":
                lo, hi = value
                lines.append(
                    f"result = lf.filter((pl.col('{column}') >= {repr(lo)}) & "
                    f"(pl.col('{column}') <= {repr(hi)}))"
                )
            # Add other filter operations as needed

        elif step.op == StepType.AGGREGATE:
            params = step.params
            groupby = params.get("groupby", [])
            metrics = params.get("metrics", [])

            agg_exprs = []
            for metric in metrics:
                col = metric["col"]
                fn = metric["fn"].lower()
                if fn == "sum":
                    agg_exprs.append(f"pl.col('{col}').sum().alias('sum_{col}')")
                elif fn == "count":
                    agg_exprs.append("pl.len().alias('count')")
                # Add other aggregations as needed

            if groupby:
                lines.append(f"result = lf.group_by({repr(groupby)}).agg([{', '.join(agg_exprs)}])")
            else:
                lines.append(f"result = lf.select([{', '.join(agg_exprs)}])")

        else:
            lines.append(f"# Operation: {step.op} with params: {step.params}")
            lines.append("result = lf  # Placeholder")

        lines.append("print(result.collect())")

        return "\n".join(lines)

    def _build_final_evidence(self, plan: PlanGraph, final_df: pl.DataFrame) -> dict[str, Any]:
        """Build comprehensive evidence for the entire execution."""
        # Convert step evidence to serializable format
        step_evidence_dicts = []
        for evidence in self.step_evidence:
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

        return {
            "plan": {
                "plan_hash": plan.plan_hash(),
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
                "rows": final_df.height,
                "columns": final_df.width,
                "column_names": final_df.columns,
            },
            "total_steps": len(self.step_evidence),
            "total_time": sum(evidence.timings["total"] for evidence in self.step_evidence),
        }


def execute(plan: PlanGraph, dataset_handle: StepHandle) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Execute a DAG plan and return results with evidence.

    Args:
        plan: DAG plan to execute
        dataset_handle: Handle to input dataset

    Returns:
        Tuple of (final_table, evidence_dict)
    """
    executor = AgentExecutor()
    return executor.execute(plan, dataset_handle)
