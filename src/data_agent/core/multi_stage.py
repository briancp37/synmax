"""Multi-stage analytical pipeline for comprehensive data analysis.

This module orchestrates multiple analytical operations and then synthesizes
results using LLM for natural language responses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ..cache import CacheManager
from .cluster import run_cluster_analysis
from .events import changepoint_detection
from .metrics import ramp_risk, reversal_freq


@dataclass
class AnalyticalResult:
    """Container for results from a single analytical operation."""

    operation: str
    data: pl.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    timing_ms: float = 0.0


@dataclass
class MultiStageResults:
    """Container for all analytical results and final synthesis."""

    original_query: str
    results: list[AnalyticalResult] = field(default_factory=list)
    synthesis: str | None = None
    llm_model_used: str | None = None
    total_timing_ms: float = 0.0


class MultiStageAnalyzer:
    """Orchestrates multiple analytical operations and LLM synthesis."""

    def __init__(self, cache_manager: CacheManager | None = None):
        self.cache_manager = cache_manager

    def analyze(
        self,
        lf: pl.LazyFrame,
        query: str,
        operations: list[str] | None = None,
        llm_model: str | None = None,
    ) -> MultiStageResults:
        """Run multi-stage analysis pipeline.

        Args:
            lf: Input data lazy frame
            query: Original natural language query
            operations: List of operations to run, or None to auto-detect
            llm_model: LLM model for synthesis

        Returns:
            MultiStageResults with all analytical outputs and synthesis
        """
        start_time = time.perf_counter()

        # Apply basic filtering based on query
        filtered_lf = self._apply_basic_filters(lf, query)

        # Auto-detect operations if not specified
        if operations is None:
            operations = self._detect_operations(query)

        results = MultiStageResults(original_query=query)

        # Run each analytical operation
        for op in operations:
            result = self._run_operation(filtered_lf, op, query)
            if result:
                results.results.append(result)

        # Synthesize results using LLM
        if results.results:
            synthesis, model_used = self._synthesize_results(results, query, llm_model)
            results.synthesis = synthesis
            results.llm_model_used = model_used

        results.total_timing_ms = (time.perf_counter() - start_time) * 1000
        return results

    def _detect_operations(self, query: str) -> list[str]:
        """Auto-detect which operations to run based on query content."""
        query_lower = query.lower()
        operations = []

        # Always include basic data summary
        operations.append("summary")

        # Changepoint detection
        if any(
            term in query_lower
            for term in ["change", "shift", "break", "discontinu", "regime", "pattern", "behavior"]
        ):
            operations.append("changepoint")

        # Clustering
        if any(
            term in query_lower
            for term in ["cluster", "group", "similar", "pattern", "behavior", "fingerprint"]
        ):
            operations.append("cluster")

        # Metrics
        if any(term in query_lower for term in ["ramp", "volatil", "swing", "variab"]):
            operations.append("ramp_risk")

        if any(
            term in query_lower
            for term in ["revers", "direction", "flow direction", "back", "forth"]
        ):
            operations.append("reversal_freq")

        if any(term in query_lower for term in ["imbalanc", "receipt", "deliver", "net", "diff"]):
            operations.append("imbalance_pct")

        # Seasonality/temporal patterns
        if any(
            term in query_lower
            for term in ["season", "month", "quarter", "annual", "cycl", "period"]
        ):
            operations.append("seasonality")

        return operations

    def _apply_basic_filters(self, lf: pl.LazyFrame, query: str) -> pl.LazyFrame:
        """Apply basic filtering based on query content."""
        query_lower = query.lower()
        filtered_lf = lf

        # Extract pipeline names mentioned in the query
        # Look for common pipeline patterns
        pipeline_patterns = [
            "anr",
            "kinder",
            "transco",
            "texas",
            "columbia",
            "williams",
            "enbridge",
            "enterprise",
            "plains",
            "magellan",
            "chevron",
            "exxon",
            "shell",
            "bp",
            "marathon",
            "valero",
        ]

        for pattern in pipeline_patterns:
            if pattern in query_lower:
                # Apply contains filter for the pipeline
                filtered_lf = filtered_lf.filter(
                    pl.col("pipeline_name").cast(pl.Utf8).str.to_lowercase().str.contains(pattern)
                )
                break  # Only apply first match

        return filtered_lf

    def _run_operation(
        self, lf: pl.LazyFrame, operation: str, query: str
    ) -> AnalyticalResult | None:
        """Run a single analytical operation."""
        start_time = time.perf_counter()

        try:
            if operation == "summary":
                return self._run_summary(lf, start_time)
            elif operation == "changepoint":
                return self._run_changepoint(lf, start_time)
            elif operation == "cluster":
                return self._run_cluster(lf, start_time)
            elif operation == "ramp_risk":
                return self._run_ramp_risk(lf, start_time)
            elif operation == "reversal_freq":
                return self._run_reversal_freq(lf, start_time)
            elif operation == "imbalance_pct":
                return self._run_imbalance_pct(lf, start_time)
            elif operation == "seasonality":
                return self._run_seasonality(lf, start_time)
            else:
                return None

        except Exception as e:
            # Return error result
            timing = (time.perf_counter() - start_time) * 1000
            return AnalyticalResult(
                operation=operation,
                data=pl.DataFrame(),
                metadata={"error": str(e)},
                timing_ms=timing,
            )

    def _run_summary(self, lf: pl.LazyFrame, start_time: float) -> AnalyticalResult:
        """Generate basic data summary."""
        # Basic aggregations
        summary_df = (
            lf.group_by("pipeline_name")
            .agg(
                [
                    pl.col("scheduled_quantity").sum().alias("total_quantity"),
                    pl.col("scheduled_quantity").mean().alias("avg_quantity"),
                    pl.col("eff_gas_day").min().alias("start_date"),
                    pl.col("eff_gas_day").max().alias("end_date"),
                    pl.len().alias("record_count"),
                ]
            )
            .collect()
        )

        timing = (time.perf_counter() - start_time) * 1000
        return AnalyticalResult(
            operation="summary",
            data=summary_df,
            metadata={"description": "Basic pipeline summary statistics"},
            timing_ms=timing,
        )

    def _run_changepoint(self, lf: pl.LazyFrame, start_time: float) -> AnalyticalResult:
        """Run changepoint detection."""
        changepoints_df = changepoint_detection(
            lf, groupby_cols=["pipeline_name"], min_confidence=0.3, penalty=0.5, min_size=3
        )

        timing = (time.perf_counter() - start_time) * 1000
        return AnalyticalResult(
            operation="changepoint",
            data=changepoints_df,
            metadata={"description": "Detected changepoints in pipeline flows"},
            timing_ms=timing,
        )

    def _run_cluster(self, lf: pl.LazyFrame, start_time: float) -> AnalyticalResult:
        """Run clustering analysis."""
        cluster_df = run_cluster_analysis(lf, entity_type="loc", k=6, random_state=42)

        timing = (time.perf_counter() - start_time) * 1000
        return AnalyticalResult(
            operation="cluster",
            data=cluster_df,
            metadata={"description": "Location clustering by behavioral patterns"},
            timing_ms=timing,
        )

    def _run_ramp_risk(self, lf: pl.LazyFrame, start_time: float) -> AnalyticalResult:
        """Run ramp risk analysis."""
        ramp_df = ramp_risk(lf).collect()

        timing = (time.perf_counter() - start_time) * 1000
        return AnalyticalResult(
            operation="ramp_risk",
            data=ramp_df,
            metadata={"description": "Pipeline ramp risk analysis (flow volatility)"},
            timing_ms=timing,
        )

    def _run_reversal_freq(self, lf: pl.LazyFrame, start_time: float) -> AnalyticalResult:
        """Run reversal frequency analysis."""
        reversal_df = reversal_freq(lf).collect()

        timing = (time.perf_counter() - start_time) * 1000
        return AnalyticalResult(
            operation="reversal_freq",
            data=reversal_df,
            metadata={"description": "Flow direction reversal frequency analysis"},
            timing_ms=timing,
        )

    def _run_imbalance_pct(self, lf: pl.LazyFrame, start_time: float) -> AnalyticalResult:
        """Run imbalance percentage analysis."""
        # Compute daily imbalances
        imbalance_df = (
            lf.with_columns(
                [
                    pl.when(pl.col("rec_del_sign") == 1)
                    .then(pl.col("scheduled_quantity"))
                    .otherwise(0.0)
                    .alias("receipts"),
                    pl.when(pl.col("rec_del_sign") == -1)
                    .then(pl.col("scheduled_quantity"))
                    .otherwise(0.0)
                    .alias("deliveries"),
                ]
            )
            .group_by(["pipeline_name", "eff_gas_day"])
            .agg(
                [
                    pl.col("receipts").sum(),
                    pl.col("deliveries").sum(),
                ]
            )
            .with_columns(
                [
                    (pl.col("receipts") - pl.col("deliveries")).alias("net_flow"),
                    (
                        pl.when(pl.col("receipts") > 0)
                        .then(
                            (pl.col("receipts") - pl.col("deliveries")) / pl.col("receipts") * 100
                        )
                        .otherwise(0.0)
                    ).alias("imbalance_pct"),
                ]
            )
            .group_by("pipeline_name")
            .agg(
                [
                    pl.col("imbalance_pct").mean().alias("avg_imbalance_pct"),
                    pl.col("imbalance_pct").std().alias("std_imbalance_pct"),
                    pl.len().alias("days_analyzed"),
                ]
            )
            .collect()
        )

        timing = (time.perf_counter() - start_time) * 1000
        return AnalyticalResult(
            operation="imbalance_pct",
            data=imbalance_df,
            metadata={"description": "Receipt/delivery imbalance analysis"},
            timing_ms=timing,
        )

    def _run_seasonality(self, lf: pl.LazyFrame, start_time: float) -> AnalyticalResult:
        """Run seasonality analysis."""
        seasonal_df = (
            lf.with_columns(
                [
                    pl.col("eff_gas_day").dt.month().alias("month"),
                    pl.col("eff_gas_day").dt.quarter().alias("quarter"),
                ]
            )
            .group_by(["pipeline_name", "month"])
            .agg(
                [
                    pl.col("scheduled_quantity").mean().alias("avg_monthly_flow"),
                    pl.col("scheduled_quantity").std().alias("std_monthly_flow"),
                    pl.len().alias("record_count"),
                ]
            )
            .collect()
        )

        timing = (time.perf_counter() - start_time) * 1000
        return AnalyticalResult(
            operation="seasonality",
            data=seasonal_df,
            metadata={"description": "Seasonal flow pattern analysis"},
            timing_ms=timing,
        )

    def _synthesize_results(
        self, results: MultiStageResults, query: str, llm_model: str | None = None
    ) -> tuple[str | None, str | None]:
        """Synthesize all analytical results using LLM."""
        try:
            # Build comprehensive context for LLM
            context = self._build_llm_context(results, query)

            # Generate synthesis using LLM
            return self._call_llm_for_synthesis(context, query, llm_model)

        except Exception as e:
            return f"Error generating synthesis: {str(e)}", None

    def _build_llm_context(self, results: MultiStageResults, query: str) -> dict[str, Any]:
        """Build structured context for LLM synthesis."""
        context = {"original_query": query, "analytical_results": {}, "summary_statistics": {}}

        for result in results.results:
            if result.data.height > 0:
                # Convert data to summary format for LLM
                if result.operation == "summary":
                    context["summary_statistics"] = result.data.to_dicts()
                elif result.operation == "changepoint":
                    context["analytical_results"]["changepoints"] = {
                        "count": result.data.height,
                        "top_changes": result.data.head(10).to_dicts(),
                        "description": result.metadata.get("description", ""),
                    }
                elif result.operation == "cluster":
                    context["analytical_results"]["clusters"] = {
                        "count": result.data.height,
                        "cluster_summary": result.data.head(10).to_dicts(),
                        "description": result.metadata.get("description", ""),
                    }
                elif result.operation in ["ramp_risk", "reversal_freq", "imbalance_pct"]:
                    context["analytical_results"][result.operation] = {
                        "results": result.data.to_dicts(),
                        "description": result.metadata.get("description", ""),
                    }
                elif result.operation == "seasonality":
                    context["analytical_results"]["seasonality"] = {
                        "monthly_patterns": result.data.to_dicts(),
                        "description": result.metadata.get("description", ""),
                    }
            else:
                # Handle empty results
                context["analytical_results"][result.operation] = {
                    "error": result.metadata.get("error", "No data found"),
                    "description": result.metadata.get("description", ""),
                }

        return context

    def _call_llm_for_synthesis(
        self, context: dict[str, Any], query: str, llm_model: str | None = None
    ) -> tuple[str | None, str | None]:
        """Call LLM to synthesize analytical results."""
        try:
            from data_agent.config import DEFAULT_CAUSAL_LLM, LLM_MODELS

            # Determine which LLM to use
            if not llm_model:
                llm_model = DEFAULT_CAUSAL_LLM

            if llm_model not in LLM_MODELS:
                return "No LLM available for synthesis", None

            model_config = LLM_MODELS[llm_model]
            provider = model_config["provider"]
            model_name = model_config["model"]

            if provider == "openai":
                return self._call_openai_synthesis(context, query, model_name)
            elif provider == "anthropic":
                return self._call_anthropic_synthesis(context, query, model_name)
            else:
                return "Unsupported LLM provider", None

        except Exception as e:
            return f"LLM synthesis failed: {str(e)}", None

    def _call_openai_synthesis(
        self, context: dict[str, Any], query: str, model: str
    ) -> tuple[str | None, str | None]:
        """Call OpenAI for result synthesis."""
        try:
            import openai

            from data_agent.config import OPENAI_API_KEY

            if not OPENAI_API_KEY:
                return "OpenAI API key not available", None

            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            # Build comprehensive prompt
            prompt = self._build_synthesis_prompt(context, query)

            # Set temperature based on model
            temperature = 1.0 if "gpt-5" in model.lower() else 0.7

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert data analyst specializing in gas pipeline ops. "
                            "Analyze the provided analytical results and provide a comprehensive, "
                            "insightful response to the user's question. Be specific, cite the "
                            "data, and provide actionable insights."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=2000,
            )

            synthesis = response.choices[0].message.content
            return synthesis, f"openai-{model}"

        except Exception as e:
            return f"OpenAI synthesis error: {str(e)}", None

    def _call_anthropic_synthesis(
        self, context: dict[str, Any], query: str, model: str
    ) -> tuple[str | None, str | None]:
        """Call Anthropic for result synthesis."""
        try:
            import anthropic

            from data_agent.config import ANTHROPIC_API_KEY

            if not ANTHROPIC_API_KEY:
                return "Anthropic API key not available", None

            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            prompt = self._build_synthesis_prompt(context, query)

            message = client.messages.create(
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are an expert data analyst specializing in gas pipelines.
Analyze the provided analytical results and provide a comprehensive, insightful response
to the user's question. Be specific, cite the data, and provide actionable insights.

{prompt}""",
                    }
                ],
                model=model,
            )

            synthesis = message.content[0].text if message.content else None
            return synthesis, f"anthropic-{model}"

        except Exception as e:
            return f"Anthropic synthesis error: {str(e)}", None

    def _build_synthesis_prompt(self, context: dict[str, Any], query: str) -> str:
        """Build comprehensive prompt for LLM synthesis."""
        import json

        prompt = f"""
ORIGINAL QUERY: {query}

ANALYTICAL RESULTS:
{json.dumps(context, indent=2, default=str)}

Please provide a comprehensive analysis that:
1. Directly answers the user's question
2. Highlights the most significant findings from each analysis
3. Explains any patterns, anomalies, or trends discovered
4. Provides specific data points and numbers to support conclusions
5. Offers actionable insights or recommendations where appropriate
6. Connects findings across different analytical dimensions

Be specific, data-driven, and insightful in your response.
"""
        return prompt
