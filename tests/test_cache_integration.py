"""Integration tests for caching with executor."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl

from data_agent.cache import CacheManager
from data_agent.core.executor import run
from data_agent.core.plan_schema import Filter, Plan


class TestCacheIntegration:
    """Test cache integration with executor."""

    def test_executor_cache_hit(self):
        """Test that executor returns cached results on repeated queries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            # Create test data
            test_data = pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["a", "b", "c", "d", "e"],
                    "value": [10, 20, 30, 40, 50],
                }
            )

            # Create a LazyFrame
            lf = test_data.lazy()

            # Create a simple plan
            plan = Plan(
                filters=[Filter(column="col1", op="=", value=3)],
                aggregate={"groupby": [], "metrics": [{"col": "value", "fn": "sum"}]},
            )

            # Mock the dataset digest and rules engine
            with patch("data_agent.core.executor._digest", return_value="test_digest"), patch(
                "data_agent.core.evidence.run_rules", return_value={}
            ):
                # First execution - should be cache miss
                answer1 = run(lf, plan, cache_manager)

                # Second execution - should be cache hit
                answer2 = run(lf, plan, cache_manager)

                # Results should be identical
                assert answer1.table.equals(answer2.table)
                assert answer1.evidence["rows_out"] == answer2.evidence["rows_out"]

                # First should be cache miss, second should be cache hit
                assert answer1.evidence["cache"]["hit"] is False
                assert answer2.evidence["cache"]["hit"] is True

                # Second execution should be faster (though we can't easily test this)
                # The evidence should show cache hit status

    def test_executor_cache_miss_different_plan(self):
        """Test that different plans result in cache misses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            # Create test data
            test_data = pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["a", "b", "c", "d", "e"],
                    "value": [10, 20, 30, 40, 50],
                }
            )

            lf = test_data.lazy()

            # Create two different plans
            plan1 = Plan(filters=[Filter(column="col1", op="=", value=2)])
            plan2 = Plan(filters=[Filter(column="col1", op="=", value=3)])

            with patch("data_agent.core.executor._digest", return_value="test_digest"), patch(
                "data_agent.core.evidence.run_rules", return_value={}
            ):
                # Execute first plan
                answer1 = run(lf, plan1, cache_manager)
                assert answer1.evidence["cache"]["hit"] is False

                # Execute second plan - should be cache miss due to different plan
                answer2 = run(lf, plan2, cache_manager)
                assert answer2.evidence["cache"]["hit"] is False

                # Results should be different
                assert not answer1.table.equals(answer2.table)

    def test_executor_cache_miss_different_dataset(self):
        """Test that different dataset digests result in cache misses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            # Create test data
            test_data = pl.DataFrame({"col1": [1, 2, 3, 4, 5], "value": [10, 20, 30, 40, 50]})

            lf = test_data.lazy()
            plan = Plan(filters=[Filter(column="col1", op="=", value=2)])

            # First execution with one digest
            with patch("data_agent.core.executor._digest", return_value="digest1"), patch(
                "data_agent.core.evidence.run_rules", return_value={}
            ):
                answer1 = run(lf, plan, cache_manager)
                assert answer1.evidence["cache"]["hit"] is False

            # Second execution with different digest - should be cache miss
            with patch("data_agent.core.executor._digest", return_value="digest2"), patch(
                "data_agent.core.evidence.run_rules", return_value={}
            ):
                answer2 = run(lf, plan, cache_manager)
                # Should be cache miss because different dataset digest
                assert answer2.evidence["cache"]["hit"] is False

    def test_executor_no_cache_manager(self):
        """Test that executor works without cache manager."""
        # Create test data
        test_data = pl.DataFrame({"col1": [1, 2, 3, 4, 5], "value": [10, 20, 30, 40, 50]})

        lf = test_data.lazy()
        plan = Plan(filters=[Filter(column="col1", op="=", value=2)])

        # Should work without cache manager
        with patch("data_agent.core.evidence.run_rules", return_value={}):
            answer = run(lf, plan, None)
        assert answer.table is not None
        assert answer.evidence is not None
        assert answer.evidence["cache"]["hit"] is False

    def test_cache_evidence_metadata(self):
        """Test that cached evidence includes proper cache metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            test_data = pl.DataFrame({"col1": [1, 2, 3], "value": [10, 20, 30]})

            lf = test_data.lazy()
            plan = Plan(filters=[Filter(column="col1", op="=", value=1)])

            with patch("data_agent.core.executor._digest", return_value="test_digest"), patch(
                "data_agent.core.evidence.run_rules", return_value={}
            ):
                # First execution
                answer1 = run(lf, plan, cache_manager)
                assert answer1.evidence["cache"]["hit"] is False

                # Second execution - should have cache metadata
                answer2 = run(lf, plan, cache_manager)
                assert answer2.evidence["cache"]["hit"] is True

                # Check that cache metadata is present
                cache_info = answer2.evidence["cache"]
                assert "fingerprint" in cache_info
                assert "ttl_hours" in cache_info
                assert "cached_at" in cache_info
