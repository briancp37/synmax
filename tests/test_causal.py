"""Tests for causal hypothesis generation module."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import polars as pl

from data_agent.core.causal import (
    CausalEvidence,
    build_causal_evidence,
    draft_hypotheses,
)
from data_agent.core.plan_schema import Plan


class TestCausalEvidence:
    """Test CausalEvidence dataclass."""

    def test_causal_evidence_creation(self):
        """Test creating CausalEvidence instance."""
        evidence = CausalEvidence(
            stats={"z_score_current": 2.5},
            peers={"percentile_total": 85.0},
            balance={"imbalance_pct_mean": 5.2},
            counterparties={"hhi": 0.25},
            calendar={"weekend_effect_pct": -10.0},
        )

        assert evidence.stats["z_score_current"] == 2.5
        assert evidence.peers["percentile_total"] == 85.0
        assert evidence.balance["imbalance_pct_mean"] == 5.2
        assert evidence.counterparties["hhi"] == 0.25
        assert evidence.calendar["weekend_effect_pct"] == -10.0

    def test_causal_evidence_to_dict(self):
        """Test converting CausalEvidence to dictionary."""
        evidence = CausalEvidence(
            stats={"z_score_current": 2.5},
            peers={"percentile_total": 85.0},
            balance={"imbalance_pct_mean": 5.2},
            counterparties={"hhi": 0.25},
            calendar={"weekend_effect_pct": -10.0},
        )

        result = evidence.to_dict()
        expected_keys = [
            "seasonal_stats",
            "peer_comparison",
            "balance_analysis",
            "counterparty_analysis",
            "calendar_signals",
        ]

        assert all(key in result for key in expected_keys)
        assert result["seasonal_stats"]["z_score_current"] == 2.5
        assert result["peer_comparison"]["percentile_total"] == 85.0


class TestBuildCausalEvidence:
    """Test build_causal_evidence function."""

    def create_test_dataframe(self) -> pl.LazyFrame:
        """Create a test dataframe for causal analysis."""
        import datetime as dt

        data = {
            "eff_gas_day": [
                dt.date(2022, 1, 1),
                dt.date(2022, 1, 2),
                dt.date(2022, 1, 3),
                dt.date(2022, 1, 4),
                dt.date(2022, 1, 5),
                dt.date(2022, 1, 6),  # Saturday
                dt.date(2022, 1, 7),  # Sunday
                dt.date(2022, 1, 8),
                dt.date(2022, 1, 9),
                dt.date(2022, 1, 10),
            ],
            "pipeline_name": ["ANR"] * 10,
            "state_abb": ["TX"] * 10,
            "scheduled_quantity": [
                100.0,
                110.0,
                105.0,
                120.0,
                115.0,
                80.0,
                85.0,
                125.0,
                130.0,
                135.0,
            ],
            "rec_del_sign": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            "connecting_entity": [
                "Entity_A",
                "Entity_B",
                "Entity_A",
                "Entity_B",
                "Entity_A",
                "Entity_B",
                "Entity_A",
                "Entity_B",
                "Entity_A",
                "Entity_B",
            ],
        }
        return pl.DataFrame(data).lazy()

    def test_build_causal_evidence_basic(self):
        """Test basic causal evidence building."""
        lf = self.create_test_dataframe()
        plan = Plan(op="causal", op_args={"target": "volume", "explain_scope": "pipeline"})

        evidence = build_causal_evidence(lf, plan)

        assert isinstance(evidence, CausalEvidence)
        assert "z_score_current" in evidence.stats
        assert "percentile_total" in evidence.peers
        assert "imbalance_pct_mean" in evidence.balance
        assert "hhi" in evidence.counterparties
        assert "weekend_effect_pct" in evidence.calendar

    def test_build_causal_evidence_empty_dataframe(self):
        """Test causal evidence building with empty dataframe."""
        lf = pl.DataFrame(
            {
                "eff_gas_day": [],
                "scheduled_quantity": [],
                "rec_del_sign": [],
                "state_abb": [],
                "connecting_entity": [],
            }
        ).lazy()

        plan = Plan(op="causal", op_args={"target": "volume"})

        evidence = build_causal_evidence(lf, plan)

        # Should handle empty data gracefully
        assert evidence.stats["z_score_current"] == 0.0
        assert evidence.peers["percentile_total"] == 50.0
        assert evidence.balance["imbalance_pct_mean"] == 0.0
        assert evidence.counterparties["hhi"] == 0.0
        assert evidence.calendar["weekend_effect_pct"] == 0.0

    def test_build_causal_evidence_different_targets(self):
        """Test causal evidence building with different targets."""
        lf = self.create_test_dataframe()

        # Test volume target
        plan_volume = Plan(op="causal", op_args={"target": "volume"})
        evidence_volume = build_causal_evidence(lf, plan_volume)
        assert isinstance(evidence_volume, CausalEvidence)

        # Test imbalance target
        plan_imbalance = Plan(op="causal", op_args={"target": "imbalance"})
        evidence_imbalance = build_causal_evidence(lf, plan_imbalance)
        assert isinstance(evidence_imbalance, CausalEvidence)

        # Both should have same structure but potentially different values
        assert evidence_volume.stats.keys() == evidence_imbalance.stats.keys()

    def test_build_causal_evidence_different_scopes(self):
        """Test causal evidence building with different scopes."""
        lf = self.create_test_dataframe()

        # Test pipeline scope
        plan_pipeline = Plan(op="causal", op_args={"explain_scope": "pipeline"})
        evidence_pipeline = build_causal_evidence(lf, plan_pipeline)
        assert evidence_pipeline.peers["scope"] == "pipeline"

        # Test state scope
        plan_state = Plan(op="causal", op_args={"explain_scope": "state"})
        evidence_state = build_causal_evidence(lf, plan_state)
        assert evidence_state.peers["scope"] == "state"


class TestDraftHypotheses:
    """Test draft_hypotheses function."""

    def create_test_evidence(self) -> CausalEvidence:
        """Create test causal evidence."""
        return CausalEvidence(
            stats={
                "z_score_current": 2.5,
                "z_score_p95": 1.8,
                "trend_direction": "increasing",
                "data_points": 100,
            },
            peers={
                "percentile_total": 85.0,
                "percentile_avg": 75.0,
                "peer_count": 10,
                "scope": "state",
            },
            balance={
                "imbalance_pct_mean": 5.2,
                "imbalance_pct_std": 2.1,
                "days_analyzed": 30,
            },
            counterparties={
                "hhi": 0.25,
                "top_3_share": 65.0,
                "counterparty_count": 8,
            },
            calendar={
                "weekend_effect_pct": -10.0,
                "weekend_avg": 90.0,
                "weekday_avg": 100.0,
                "weekend_days": 8,
                "weekday_days": 22,
            },
        )

    def test_draft_hypotheses_no_llm(self):
        """Test hypothesis drafting without LLM client."""
        evidence = self.create_test_evidence()

        hypotheses = draft_hypotheses(evidence, llm_client=None)

        # Without LLM, should return empty list
        assert hypotheses == []

    @patch("data_agent.core.causal._get_llm_client")
    def test_draft_hypotheses_no_available_llm(self, mock_get_client):
        """Test hypothesis drafting when no LLM is available."""
        mock_get_client.return_value = None
        evidence = self.create_test_evidence()

        hypotheses = draft_hypotheses(evidence)

        # Should return empty list when no LLM available
        assert hypotheses == []

    @patch("data_agent.core.causal._call_openai_for_hypotheses")
    def test_draft_hypotheses_with_openai_mock(self, mock_openai):
        """Test hypothesis drafting with mocked OpenAI client."""
        # Mock OpenAI response
        mock_hypotheses = [
            {
                "mechanism": "Seasonal demand increase drove higher volumes",
                "evidence": [
                    "Z-score of 2.5 indicates significant deviation from baseline",
                    "85th percentile vs peers suggests above-average performance",
                ],
                "confidence": "medium",
                "caveats": [
                    "Limited to 30-day analysis period",
                    "External factors not considered",
                ],
            },
            {
                "mechanism": "Market concentration changes affected flow patterns",
                "evidence": [
                    "HHI of 0.25 indicates moderate market concentration",
                    "Top 3 entities control 65% of market share",
                ],
                "confidence": "low",
                "caveats": [
                    "Correlation does not imply causation",
                    "Need longer time series for validation",
                ],
            },
        ]
        mock_openai.return_value = mock_hypotheses

        # Create mock client
        mock_client = Mock()
        mock_client.chat = Mock()  # OpenAI client signature

        evidence = self.create_test_evidence()
        hypotheses = draft_hypotheses(evidence, llm_client=mock_client)

        assert len(hypotheses) == 2
        assert hypotheses[0]["mechanism"] == "Seasonal demand increase drove higher volumes"
        assert hypotheses[0]["confidence"] == "medium"
        assert len(hypotheses[0]["evidence"]) == 2
        assert len(hypotheses[0]["caveats"]) == 2

        # Verify OpenAI function was called
        mock_openai.assert_called_once()

    def test_draft_hypotheses_with_anthropic_mock(self):
        """Test hypothesis drafting with mocked Anthropic client."""
        # Mock Anthropic response
        mock_hypotheses = [
            {
                "mechanism": "Weekend scheduling patterns created imbalance",
                "evidence": [
                    "Weekend effect of -10% indicates reduced activity",
                    "Imbalance of 5.2% above typical thresholds",
                ],
                "confidence": "high",
                "caveats": [
                    "Limited to observed time period",
                ],
            }
        ]

        evidence = self.create_test_evidence()

        # Mock both the _call_openai_for_hypotheses (to return None) and
        # _call_anthropic_for_hypotheses
        with patch("data_agent.core.causal._call_openai_for_hypotheses", return_value=None):
            with patch(
                "data_agent.core.causal._call_anthropic_for_hypotheses",
                return_value=mock_hypotheses,
            ):
                # Create mock client that has messages attribute but NOT chat (Anthropic signature)
                mock_client = Mock()
                mock_client.messages = Mock()
                # Ensure it doesn't have chat attribute to avoid OpenAI path
                if hasattr(mock_client, "chat"):
                    delattr(mock_client, "chat")

                hypotheses = draft_hypotheses(evidence, llm_client=mock_client)

        assert len(hypotheses) == 1
        assert hypotheses[0]["mechanism"] == "Weekend scheduling patterns created imbalance"
        assert hypotheses[0]["confidence"] == "high"

    def test_draft_hypotheses_llm_failure(self):
        """Test hypothesis drafting when LLM calls fail."""
        # Create mock client that will cause failures
        mock_client = Mock()
        mock_client.chat = Mock()

        with patch("data_agent.core.causal._call_openai_for_hypotheses", return_value=None):
            with patch("data_agent.core.causal._call_anthropic_for_hypotheses", return_value=None):
                evidence = self.create_test_evidence()
                hypotheses = draft_hypotheses(evidence, llm_client=mock_client)

                # Should return empty list when LLM calls fail
                assert hypotheses == []


class TestIntegration:
    """Integration tests for causal module."""

    def test_full_causal_pipeline(self):
        """Test the complete causal analysis pipeline."""
        import datetime as dt

        # Create realistic test data
        data = {
            "eff_gas_day": [dt.date(2022, 1, i) for i in range(1, 31)],
            "pipeline_name": ["ANR"] * 30,
            "state_abb": ["TX"] * 30,
            "scheduled_quantity": [
                100 + i * 2 + (i % 7) * 10 for i in range(30)
            ],  # Trend + weekly pattern
            "rec_del_sign": [1 if i % 2 == 0 else -1 for i in range(30)],
            "connecting_entity": [f"Entity_{i % 5}" for i in range(30)],
        }
        lf = pl.DataFrame(data).lazy()

        plan = Plan(
            op="causal",
            op_args={
                "target": "volume",
                "explain_scope": "state",
                "compare_to": "prior_period",
                "top_k_factors": 5,
            },
        )

        # Build evidence
        evidence = build_causal_evidence(lf, plan)

        # Verify evidence structure
        assert isinstance(evidence, CausalEvidence)
        assert evidence.stats["data_points"] == 30
        assert evidence.peers["scope"] == "state"
        assert evidence.counterparties["counterparty_count"] == 5

        # Test evidence to dict conversion
        evidence_dict = evidence.to_dict()
        assert "seasonal_stats" in evidence_dict
        assert "peer_comparison" in evidence_dict
        assert "balance_analysis" in evidence_dict
        assert "counterparty_analysis" in evidence_dict
        assert "calendar_signals" in evidence_dict

        # Test JSON serialization (important for LLM integration)
        evidence_json = json.dumps(evidence_dict)
        assert isinstance(evidence_json, str)
        assert len(evidence_json) > 0

        # Parse back to verify it's valid JSON
        parsed = json.loads(evidence_json)
        assert parsed["seasonal_stats"]["data_points"] == 30

    def test_causal_evidence_with_minimal_data(self):
        """Test causal evidence building with minimal data."""
        import datetime as dt

        # Create slightly more data to avoid Polars groupby issues
        data = {
            "eff_gas_day": [dt.date(2022, 1, 1), dt.date(2022, 1, 2)],
            "scheduled_quantity": [100.0, 110.0],
            "rec_del_sign": [1, -1],
            "state_abb": ["TX", "TX"],
            "connecting_entity": ["Entity_A", "Entity_A"],
        }
        lf = pl.DataFrame(data).lazy()

        plan = Plan(op="causal")
        evidence = build_causal_evidence(lf, plan)

        # Should handle minimal data without errors
        assert isinstance(evidence, CausalEvidence)
        # With minimal data, some computations might return 0 or fallback values
        assert evidence.stats["data_points"] >= 0  # Should be non-negative
        assert evidence.counterparties["counterparty_count"] >= 0  # Should be non-negative
