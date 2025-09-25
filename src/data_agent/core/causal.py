"""Causal hypothesis generation module.

This module computes diagnostic features for causal analysis and generates
plausible hypotheses using LLM integration when available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import polars as pl

from .plan_schema import Plan


@dataclass
class CausalEvidence:
    """Structured evidence for causal hypothesis generation."""

    stats: dict[str, Any]
    peers: dict[str, Any]
    balance: dict[str, Any]
    counterparties: dict[str, Any]
    calendar: dict[str, Any]
    multidim: dict[str, Any] = None  # For multi-dimensional analysis
    data_type: str = "pipeline"  # "pipeline" or "changepoint" or "cluster"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM consumption."""
        result = {
            "data_type": self.data_type,
            "seasonal_stats": self.stats,
            "peer_comparison": self.peers,
            "balance_analysis": self.balance,
            "counterparty_analysis": self.counterparties,
            "calendar_signals": self.calendar,
        }
        if self.multidim:
            result["multidimensional_analysis"] = self.multidim
        return result


def _compute_seasonal_deviation(
    lf: pl.LazyFrame, value_col: str = "scheduled_quantity"
) -> dict[str, Any]:
    """Compute seasonal deviation metrics with detrended z-scores."""
    try:
        # Compute rolling 30-day mean and z-score
        df = (
            lf.select(
                [
                    pl.col("eff_gas_day"),
                    pl.col(value_col),
                ]
            )
            .sort("eff_gas_day")
            .with_columns(
                [
                    # Rolling 30-day mean (detrending)
                    pl.col(value_col).rolling_mean(window_size=30).alias("rolling_mean"),
                    # Overall mean and std for z-score
                    pl.col(value_col).mean().alias("overall_mean"),
                    pl.col(value_col).std().alias("overall_std"),
                ]
            )
            .with_columns(
                [
                    # Detrended value
                    (pl.col(value_col) - pl.col("rolling_mean")).alias("detrended"),
                ]
            )
            .with_columns(
                [
                    # Z-score of detrended values
                    (pl.col("detrended") / pl.col("overall_std")).alias("z_score"),
                ]
            )
            .collect()
        )

        if df.height == 0:
            return {"z_score_current": 0.0, "z_score_p95": 0.0, "trend_direction": "stable"}

        # Get current period stats (last 30 days or all available data)
        recent_df = df.tail(min(30, df.height))
        current_z = (
            float(recent_df.select(pl.col("z_score").mean()).item())
            if recent_df.height > 0
            else 0.0
        )

        # Overall z-score percentiles
        z_p95 = float(df.select(pl.col("z_score").quantile(0.95)).item()) if df.height > 0 else 0.0

        # Trend direction based on recent slope
        trend = "stable"
        if df.height >= 7:
            recent_values = df.tail(7).select(pl.col(value_col)).to_numpy().flatten()
            if len(recent_values) >= 2:
                slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
                if slope > 0.1 * recent_values.mean():
                    trend = "increasing"
                elif slope < -0.1 * recent_values.mean():
                    trend = "decreasing"

        return {
            "z_score_current": round(current_z, 2),
            "z_score_p95": round(z_p95, 2),
            "trend_direction": trend,
            "data_points": int(df.height),
        }
    except Exception:
        return {
            "z_score_current": 0.0,
            "z_score_p95": 0.0,
            "trend_direction": "stable",
            "data_points": 0,
        }


def _compute_peer_contrast(lf: pl.LazyFrame, scope: str = "state") -> dict[str, Any]:
    """Compute percentile rank vs peers."""
    try:
        if scope == "state":
            group_col = "state_abb"
        elif scope == "pipeline":
            group_col = "pipeline_name"
        else:
            group_col = "state_abb"  # default fallback

        # Compute total scheduled quantity by group
        df = (
            lf.group_by(group_col)
            .agg(
                [
                    pl.col("scheduled_quantity").sum().alias("total_volume"),
                    pl.col("scheduled_quantity").mean().alias("avg_volume"),
                ]
            )
            .with_columns(
                [
                    # Compute percentile ranks
                    pl.col("total_volume").rank(method="average").alias("total_rank"),
                    pl.col("avg_volume").rank(method="average").alias("avg_rank"),
                ]
            )
            .with_columns(
                [
                    # Convert to percentiles (0-100)
                    (pl.col("total_rank") / pl.len() * 100).alias("total_percentile"),
                    (pl.col("avg_rank") / pl.len() * 100).alias("avg_percentile"),
                ]
            )
            .collect()
        )

        if df.height == 0:
            return {"percentile_total": 50.0, "percentile_avg": 50.0, "peer_count": 0}

        # Get current entity's percentile (take first as example)
        total_pct = float(df.select(pl.col("total_percentile").mean()).item())
        avg_pct = float(df.select(pl.col("avg_percentile").mean()).item())

        return {
            "percentile_total": round(total_pct, 1),
            "percentile_avg": round(avg_pct, 1),
            "peer_count": int(df.height),
            "scope": scope,
        }
    except Exception:
        return {"percentile_total": 50.0, "percentile_avg": 50.0, "peer_count": 0, "scope": scope}


def _compute_balance_residual(lf: pl.LazyFrame) -> dict[str, Any]:
    """Compute imbalance percentage contemporaneous."""
    try:
        df = (
            lf.with_columns(
                [
                    # Create receipt and delivery columns
                    pl.when(pl.col("rec_del_sign") == 1)
                    .then(pl.col("scheduled_quantity"))
                    .otherwise(0.0)
                    .alias("receipt_qty"),
                    pl.when(pl.col("rec_del_sign") == -1)
                    .then(pl.col("scheduled_quantity"))
                    .otherwise(0.0)
                    .alias("delivery_qty"),
                ]
            )
            .group_by("eff_gas_day")
            .agg(
                [
                    pl.col("receipt_qty").sum().alias("receipts"),
                    pl.col("delivery_qty").sum().alias("deliveries"),
                ]
            )
            .with_columns(
                [
                    # Net flow and imbalance percentage
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
            .collect()
        )

        if df.height == 0:
            return {"imbalance_pct_mean": 0.0, "imbalance_pct_std": 0.0, "days_analyzed": 0}

        imbalance_mean = float(df.select(pl.col("imbalance_pct").mean()).item())
        imbalance_std = float(df.select(pl.col("imbalance_pct").std()).item() or 0.0)

        return {
            "imbalance_pct_mean": round(imbalance_mean, 2),
            "imbalance_pct_std": round(imbalance_std, 2),
            "days_analyzed": int(df.height),
        }
    except Exception:
        return {"imbalance_pct_mean": 0.0, "imbalance_pct_std": 0.0, "days_analyzed": 0}


def _compute_counterparty_shift(lf: pl.LazyFrame) -> dict[str, Any]:
    """Compute HHI change and top contributor deltas."""
    try:
        # Compute volume by counterparty (using connecting_entity)
        df = (
            lf.group_by("connecting_entity")
            .agg(
                [
                    pl.col("scheduled_quantity").sum().alias("total_volume"),
                ]
            )
            .filter(pl.col("connecting_entity").is_not_null())
            .with_columns(
                [
                    # Market share
                    (pl.col("total_volume") / pl.col("total_volume").sum()).alias("market_share"),
                ]
            )
            .with_columns(
                [
                    # Squared market share for HHI
                    (pl.col("market_share") ** 2).alias("share_squared"),
                ]
            )
            .sort("total_volume", descending=True)
            .collect()
        )

        if df.height == 0:
            return {"hhi": 0.0, "top_3_share": 0.0, "counterparty_count": 0}

        # Compute HHI (Herfindahl-Hirschman Index)
        hhi = float(df.select(pl.col("share_squared").sum()).item())

        # Top 3 market share
        top_3_share = float(df.head(3).select(pl.col("market_share").sum()).item())

        return {
            "hhi": round(hhi, 4),
            "top_3_share": round(top_3_share * 100, 2),  # Convert to percentage
            "counterparty_count": int(df.height),
        }
    except Exception:
        return {"hhi": 0.0, "top_3_share": 0.0, "counterparty_count": 0}


def _compute_calendar_signals(lf: pl.LazyFrame) -> dict[str, Any]:
    """Compute calendar signals (weekend proximity only)."""
    try:
        df = (
            lf.select(
                [
                    pl.col("eff_gas_day"),
                    pl.col("scheduled_quantity"),
                ]
            )
            .with_columns(
                [
                    # Day of week (Monday = 1, Sunday = 7)
                    pl.col("eff_gas_day")
                    .dt.weekday()
                    .alias("day_of_week"),
                ]
            )
            .with_columns(
                [
                    # Weekend indicator
                    (pl.col("day_of_week").is_in([6, 7])).alias("is_weekend"),
                ]
            )
            .group_by("is_weekend")
            .agg(
                [
                    pl.col("scheduled_quantity").mean().alias("avg_volume"),
                    pl.len().alias("day_count"),
                ]
            )
            .collect()
        )

        weekend_avg = 0.0
        weekday_avg = 0.0
        weekend_days = 0
        weekday_days = 0

        for row in df.iter_rows(named=True):
            if row["is_weekend"]:
                weekend_avg = float(row["avg_volume"] or 0.0)
                weekend_days = int(row["day_count"])
            else:
                weekday_avg = float(row["avg_volume"] or 0.0)
                weekday_days = int(row["day_count"])

        # Weekend effect (percentage difference)
        weekend_effect = 0.0
        if weekday_avg > 0:
            weekend_effect = ((weekend_avg - weekday_avg) / weekday_avg) * 100

        return {
            "weekend_effect_pct": round(weekend_effect, 2),
            "weekend_avg": round(weekend_avg, 2),
            "weekday_avg": round(weekday_avg, 2),
            "weekend_days": weekend_days,
            "weekday_days": weekday_days,
        }
    except Exception:
        return {
            "weekend_effect_pct": 0.0,
            "weekend_avg": 0.0,
            "weekday_avg": 0.0,
            "weekend_days": 0,
            "weekday_days": 0,
        }


def build_causal_evidence(lf: pl.LazyFrame, plan: Plan) -> CausalEvidence:
    """Build comprehensive causal evidence from any dataset schema.

    Args:
        lf: Filtered lazy frame to analyze
        plan: Query plan containing causal arguments

    Returns:
        CausalEvidence with computed diagnostic features
    """
    # Auto-detect schema
    columns = lf.collect_schema().names()

    # Detect data type and schema
    if "changepoint_date" in columns:
        return _build_changepoint_evidence(lf, plan, columns)
    elif "cluster_id" in columns:
        return _build_cluster_evidence(lf, plan, columns)
    else:
        return _build_pipeline_evidence(lf, plan, columns)


def _build_pipeline_evidence(lf: pl.LazyFrame, plan: Plan, columns: list[str]) -> CausalEvidence:
    """Build causal evidence from original pipeline data."""

    # Extract causal arguments from plan
    causal_args = plan.op_args if plan.op_args else {}
    target = causal_args.get("target", "volume")
    scope = causal_args.get("explain_scope", "pipeline")

    # Determine value column based on target
    value_col = "scheduled_quantity"  # Default for volume
    if target == "imbalance":
        value_col = "scheduled_quantity"  # Will compute imbalance separately
    elif target in ["ramp_risk", "reversal_freq"]:
        value_col = "scheduled_quantity"  # Base column for these metrics

    try:
        # Compute diagnostic features (original logic)
        stats = _compute_seasonal_deviation(lf, value_col)
        peers = _compute_peer_contrast(lf, scope)
        balance = _compute_balance_residual(lf)
        counterparties = _compute_counterparty_shift(lf)
        calendar = _compute_calendar_signals(lf)

        return CausalEvidence(
            stats=stats,
            peers=peers,
            balance=balance,
            counterparties=counterparties,
            calendar=calendar,
            data_type="pipeline",
        )
    except Exception:
        # Fallback to basic analysis
        return _build_basic_evidence(lf, columns)


def _build_changepoint_evidence(lf: pl.LazyFrame, plan: Plan, columns: list[str]) -> CausalEvidence:
    """Build causal evidence from changepoint detection results."""

    # Multi-dimensional analysis for changepoint data
    multidim = {}

    try:
        # Analyze change magnitudes
        if "change_magnitude" in columns:
            multidim["magnitude_analysis"] = _analyze_change_magnitudes(lf)

        # Analyze confidence patterns
        if "confidence" in columns:
            multidim["confidence_analysis"] = _analyze_confidence_patterns(lf)

        # Analyze volume changes (before vs after)
        if "before_mean" in columns and "after_mean" in columns:
            multidim["volume_change_analysis"] = _analyze_volume_changes(lf)

        # Analyze volatility changes
        if "before_std" in columns and "after_std" in columns:
            multidim["volatility_analysis"] = _analyze_volatility_changes(lf)

        # Temporal clustering analysis
        if "changepoint_date" in columns:
            multidim["temporal_analysis"] = _analyze_changepoint_timing(lf)

        # Cross-dimensional correlations
        numerical_cols = [
            col
            for col in columns
            if col
            in [
                "change_magnitude",
                "confidence",
                "before_mean",
                "after_mean",
                "before_std",
                "after_std",
            ]
        ]
        if len(numerical_cols) > 1:
            multidim["correlation_analysis"] = _compute_cross_correlations(lf, numerical_cols)

        return CausalEvidence(
            stats={"data_points": lf.select(pl.count()).collect().item()},
            peers={},
            balance={},
            counterparties={},
            calendar={},
            multidim=multidim,
            data_type="changepoint",
        )
    except Exception:
        return _build_basic_evidence(lf, columns)


def _build_cluster_evidence(lf: pl.LazyFrame, plan: Plan, columns: list[str]) -> CausalEvidence:
    """Build causal evidence from clustering results."""

    # For cluster data, analyze cluster characteristics
    multidim = {}

    try:
        if "cluster_id" in columns:
            multidim["cluster_analysis"] = _analyze_cluster_patterns(lf, columns)

        return CausalEvidence(
            stats={"data_points": lf.select(pl.count()).collect().item()},
            peers={},
            balance={},
            counterparties={},
            calendar={},
            multidim=multidim,
            data_type="cluster",
        )
    except Exception:
        return _build_basic_evidence(lf, columns)


def _build_basic_evidence(lf: pl.LazyFrame, columns: list[str]) -> CausalEvidence:
    """Build basic evidence when specialized analysis fails."""

    try:
        basic_stats = {
            "data_points": lf.select(pl.count()).collect().item(),
            "columns": columns,
            "column_count": len(columns),
        }

        return CausalEvidence(
            stats=basic_stats,
            peers={},
            balance={},
            counterparties={},
            calendar={},
            data_type="unknown",
        )
    except Exception as e:
        return CausalEvidence(
            stats={"error": str(e)},
            peers={},
            balance={},
            counterparties={},
            calendar={},
            data_type="error",
        )


def _get_llm_client(llm_model: str | None = None) -> tuple[Any | None, str | None]:
    """Get LLM client for specified model.

    Args:
        llm_model: LLM model identifier (e.g., 'openai-gpt4', 'anthropic-sonnet')
                  If None, uses default from config

    Returns:
        Tuple of (client, model_name) or (None, None) if unavailable
    """
    from data_agent.config import ANTHROPIC_API_KEY, DEFAULT_CAUSAL_LLM, LLM_MODELS, OPENAI_API_KEY

    if llm_model is None:
        llm_model = DEFAULT_CAUSAL_LLM

    if llm_model not in LLM_MODELS:
        # Find the first available OpenAI model as fallback
        fallback_model = None
        for model_id in LLM_MODELS.keys():
            if model_id.startswith("openai-"):
                fallback_model = model_id
                break

        if fallback_model:
            print(f"Warning: Unknown LLM model '{llm_model}', using fallback '{fallback_model}'")
            llm_model = fallback_model
        else:
            print(f"Error: Unknown LLM model '{llm_model}' and no OpenAI fallback available")
            return None, None

    model_config = LLM_MODELS[llm_model]
    provider = model_config["provider"]
    model_name = model_config["model"]

    if provider == "openai":
        try:
            import openai

            if OPENAI_API_KEY:
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                return client, model_name
        except ImportError as e:
            print(f"Warning: OpenAI library not available: {e}")
        except Exception:
            pass

    elif provider == "anthropic":
        try:
            import anthropic

            if ANTHROPIC_API_KEY:
                return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY), model_name
        except ImportError:
            print("Warning: Anthropic library not available")

    return None, None


def _call_openai_for_hypotheses(
    evidence_json: str, client: Any, model: str = "gpt-4"
) -> list[dict[str, Any]] | None:
    """Call OpenAI to generate causal hypotheses."""
    try:
        # GPT-5 models only support default temperature (1.0)
        temperature = 1.0 if model.startswith("gpt-5") else 0.7

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a cautious energy analyst. "
                    "Propose plausible causes; never assert.",
                },
                {
                    "role": "user",
                    "content": f"""Evidence JSON: {evidence_json}

Task: Produce 3-5 hypotheses explaining the observed relationship/pattern. For each:
- Mechanism (1 sentence)
- Evidence (2-4 bullets, cite numbers from evidence JSON)
- Confidence: low|medium|high (justify briefly)
- Caveats (1-2 bullets)

Return as JSON list of objects: {{"mechanism","evidence":[],"confidence","caveats":[]}}""",
                },
            ],
            temperature=temperature,
        )

        content = response.choices[0].message.content
        if content:
            # Try to extract JSON from the response
            import re

            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if isinstance(result, list):
                    return result
    except Exception as e:
        print(f"ERROR: OpenAI API call failed with model '{model}': {type(e).__name__}: {e}")
    return None


def _call_anthropic_for_hypotheses(
    evidence_json: str, client: Any, model: str = "claude-3-sonnet-20240229"
) -> list[dict[str, Any]] | None:
    """Call Anthropic to generate causal hypotheses."""
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a cautious energy analyst. """
                    f"""Propose plausible causes; never assert.

Evidence JSON: {evidence_json}

Task: Produce 3-5 hypotheses explaining the observed relationship/pattern. For each:
- Mechanism (1 sentence)
- Evidence (2-4 bullets, cite numbers from evidence JSON)
- Confidence: low|medium|high (justify briefly)
- Caveats (1-2 bullets)

Return as JSON list of objects: {{"mechanism","evidence":[],"confidence","caveats":[]}}""",
                }
            ],
        )

        content = message.content[0].text if message.content else None
        if content:
            # Try to extract JSON from the response
            import re

            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if isinstance(result, list):
                    return result
    except Exception:
        pass
    return None


def draft_hypotheses(
    evidence: CausalEvidence, llm_model: str | None = None, llm_client: Any | None = None
) -> tuple[list[dict[str, Any]], str | None]:
    """Draft causal hypotheses using LLM or return deterministic fallback.

    Args:
        evidence: Computed causal evidence
        llm_model: LLM model identifier (e.g., 'openai-gpt4', 'anthropic-sonnet')
        llm_client: Optional LLM client (deprecated, use llm_model instead)

    Returns:
        Tuple of (hypothesis list, model_used) where model_used is the actual model identifier used
    """

    # Get client and model name
    model_used = None
    if llm_client is None:
        llm_client, model_name = _get_llm_client(llm_model)

        # Track which model config was actually used
        if llm_model is None:
            from data_agent.config import DEFAULT_CAUSAL_LLM

            model_used = DEFAULT_CAUSAL_LLM
        else:
            model_used = llm_model
    else:
        # Legacy support - if client is provided directly, use default model
        model_name = "gpt-4"  # fallback
        model_used = "unknown (legacy client)"

    if llm_client is None:
        # Deterministic fallback
        return [], None

    # Convert evidence to JSON for LLM
    evidence_json = json.dumps(evidence.to_dict(), indent=2)

    # Try to get hypotheses from LLM
    hypotheses = None

    # Check if it's OpenAI client
    if hasattr(llm_client, "chat"):
        hypotheses = _call_openai_for_hypotheses(evidence_json, llm_client, model_name or "gpt-4")
    # Check if it's Anthropic client
    elif hasattr(llm_client, "messages"):
        hypotheses = _call_anthropic_for_hypotheses(
            evidence_json, llm_client, model_name or "claude-3-sonnet-20240229"
        )
    else:
        hypotheses = None

    # Return hypotheses and model used
    return hypotheses or [], model_used


# Multi-dimensional analysis functions for changepoint data


def _analyze_change_magnitudes(lf: pl.LazyFrame) -> dict[str, Any]:
    """Analyze change magnitude patterns."""
    try:
        stats = lf.select(
            [
                pl.col("change_magnitude").mean().alias("mean_magnitude"),
                pl.col("change_magnitude").std().alias("std_magnitude"),
                pl.col("change_magnitude").min().alias("min_magnitude"),
                pl.col("change_magnitude").max().alias("max_magnitude"),
                pl.col("change_magnitude").quantile(0.5).alias("median_magnitude"),
                pl.col("change_magnitude").quantile(0.75).alias("p75_magnitude"),
                pl.col("change_magnitude").quantile(0.95).alias("p95_magnitude"),
                (pl.col("change_magnitude") > 0.5).sum().alias("high_magnitude_count"),
                pl.count().alias("total_changes"),
            ]
        ).collect()

        return stats.to_dicts()[0]
    except Exception as e:
        return {"error": str(e)}


def _analyze_confidence_patterns(lf: pl.LazyFrame) -> dict[str, Any]:
    """Analyze confidence score patterns."""
    try:
        stats = lf.select(
            [
                pl.col("confidence").mean().alias("mean_confidence"),
                pl.col("confidence").std().alias("std_confidence"),
                pl.col("confidence").min().alias("min_confidence"),
                pl.col("confidence").max().alias("max_confidence"),
                (pl.col("confidence") > 0.8).sum().alias("high_confidence_count"),
                (pl.col("confidence") < 0.5).sum().alias("low_confidence_count"),
                pl.count().alias("total_changes"),
            ]
        ).collect()

        return stats.to_dicts()[0]
    except Exception as e:
        return {"error": str(e)}


def _analyze_volume_changes(lf: pl.LazyFrame) -> dict[str, Any]:
    """Analyze before vs after volume changes."""
    try:
        stats = lf.select(
            [
                pl.col("before_mean").mean().alias("avg_before_mean"),
                pl.col("after_mean").mean().alias("avg_after_mean"),
                (pl.col("after_mean") - pl.col("before_mean")).mean().alias("avg_volume_change"),
                (pl.col("after_mean") - pl.col("before_mean")).std().alias("std_volume_change"),
                (pl.col("after_mean") > pl.col("before_mean")).sum().alias("volume_increases"),
                (pl.col("after_mean") < pl.col("before_mean")).sum().alias("volume_decreases"),
                ((pl.col("after_mean") - pl.col("before_mean")) / pl.col("before_mean") * 100)
                .mean()
                .alias("avg_pct_change"),
                pl.count().alias("total_changes"),
            ]
        ).collect()

        return stats.to_dicts()[0]
    except Exception as e:
        return {"error": str(e)}


def _analyze_volatility_changes(lf: pl.LazyFrame) -> dict[str, Any]:
    """Analyze before vs after volatility changes."""
    try:
        stats = lf.select(
            [
                pl.col("before_std").mean().alias("avg_before_volatility"),
                pl.col("after_std").mean().alias("avg_after_volatility"),
                (pl.col("after_std") - pl.col("before_std")).mean().alias("avg_volatility_change"),
                (pl.col("after_std") > pl.col("before_std")).sum().alias("volatility_increases"),
                (pl.col("after_std") < pl.col("before_std")).sum().alias("volatility_decreases"),
                pl.count().alias("total_changes"),
            ]
        ).collect()

        return stats.to_dicts()[0]
    except Exception as e:
        return {"error": str(e)}


def _analyze_changepoint_timing(lf: pl.LazyFrame) -> dict[str, Any]:
    """Analyze temporal patterns of changepoints."""
    try:
        # Monthly distribution
        monthly_stats = (
            lf.select([pl.col("changepoint_date").dt.month().alias("month")])
            .group_by("month")
            .count()
            .sort("month")
            .collect()
        )

        # Yearly distribution
        yearly_stats = (
            lf.select([pl.col("changepoint_date").dt.year().alias("year")])
            .group_by("year")
            .count()
            .sort("year")
            .collect()
        )

        # Day of week distribution
        dow_stats = (
            lf.select([pl.col("changepoint_date").dt.weekday().alias("day_of_week")])
            .group_by("day_of_week")
            .count()
            .sort("day_of_week")
            .collect()
        )

        return {
            "monthly_distribution": monthly_stats.to_dicts(),
            "yearly_distribution": yearly_stats.to_dicts(),
            "day_of_week_distribution": dow_stats.to_dicts(),
            "total_changepoints": lf.select(pl.count()).collect().item(),
            "date_range": {
                "earliest": lf.select(pl.col("changepoint_date").min()).collect().item(),
                "latest": lf.select(pl.col("changepoint_date").max()).collect().item(),
            },
        }
    except Exception as e:
        return {"error": str(e)}


def _compute_cross_correlations(lf: pl.LazyFrame, numerical_cols: list[str]) -> dict[str, Any]:
    """Compute correlations between numerical columns."""
    try:
        correlations = {}
        df = lf.collect()

        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i + 1 :]:
                try:
                    corr = df.select(pl.corr(col1, col2)).item()
                    correlations[f"{col1}_vs_{col2}"] = corr
                except Exception:
                    correlations[f"{col1}_vs_{col2}"] = None

        return {"correlations": correlations, "analyzed_columns": numerical_cols}
    except Exception as e:
        return {"error": str(e)}


def _analyze_cluster_patterns(lf: pl.LazyFrame, columns: list[str]) -> dict[str, Any]:
    """Analyze cluster characteristics."""
    try:
        # Basic cluster statistics
        cluster_stats = lf.group_by("cluster_id").count().sort("cluster_id").collect()

        return {
            "cluster_distribution": cluster_stats.to_dicts(),
            "total_clusters": cluster_stats.height,
            "total_points": lf.select(pl.count()).collect().item(),
        }
    except Exception as e:
        return {"error": str(e)}
