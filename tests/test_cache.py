"""Tests for caching functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl

from data_agent.cache import CacheManager
from data_agent.core.plan_schema import Filter, Plan


class TestCacheManager:
    """Test cache manager functionality."""

    def test_init(self):
        """Test cache manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache_manager = CacheManager(cache_dir=cache_dir, ttl_hours=1)

            assert cache_manager.cache_dir == cache_dir
            assert cache_manager.ttl_hours == 1
            assert cache_manager.ttl_seconds == 3600
            assert cache_dir.exists()

    def test_fingerprint_computation(self):
        """Test fingerprint computation from plan and dataset digest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            plan = Plan(filters=[Filter(column="test_col", op="=", value="test")])
            dataset_digest = "abc123"

            fingerprint1 = cache_manager._compute_fingerprint(plan, dataset_digest)
            fingerprint2 = cache_manager._compute_fingerprint(plan, dataset_digest)

            # Same inputs should produce same fingerprint
            assert fingerprint1 == fingerprint2
            assert len(fingerprint1) == 64  # SHA-256 hex length

            # Different plan should produce different fingerprint
            plan2 = Plan(filters=[Filter(column="other_col", op="=", value="test")])
            fingerprint3 = cache_manager._compute_fingerprint(plan2, dataset_digest)
            assert fingerprint1 != fingerprint3

            # Different dataset digest should produce different fingerprint
            fingerprint4 = cache_manager._compute_fingerprint(plan, "xyz789")
            assert fingerprint1 != fingerprint4

    def test_cache_put_and_get(self):
        """Test storing and retrieving from cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            plan = Plan(filters=[Filter(column="test_col", op="=", value="test")])
            dataset_digest = "abc123"

            # Create test data
            df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            evidence = {"test": "evidence", "cache": {"hit": False}}

            # Initially should not be in cache
            cached_df, cached_evidence = cache_manager.get(plan, dataset_digest)
            assert cached_df is None
            assert cached_evidence is None

            # Store in cache
            cache_manager.put(plan, dataset_digest, df, evidence)

            # Should now be retrievable
            cached_df, cached_evidence = cache_manager.get(plan, dataset_digest)
            assert cached_df is not None
            assert cached_evidence is not None
            assert cached_df.equals(df)

            # Check that the evidence matches (accounting for cache metadata)
            assert cached_evidence["test"] == evidence["test"]
            assert "cache" in cached_evidence
            assert "fingerprint" in cached_evidence["cache"]
            assert "ttl_hours" in cached_evidence["cache"]
            assert "cached_at" in cached_evidence["cache"]

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use very short TTL for testing
            cache_manager = CacheManager(
                cache_dir=Path(temp_dir) / "cache", ttl_hours=0.0001
            )  # ~0.36 seconds

            plan = Plan(filters=[Filter(column="test_col", op="=", value="test")])
            dataset_digest = "abc123"

            df = pl.DataFrame({"col1": [1, 2, 3]})
            evidence = {"test": "evidence"}

            # Store in cache
            cache_manager.put(plan, dataset_digest, df, evidence)

            # Should be available immediately
            cached_df, cached_evidence = cache_manager.get(plan, dataset_digest)
            assert cached_df is not None

            # Wait for TTL to expire
            import time

            time.sleep(0.5)

            # Should no longer be available
            cached_df, cached_evidence = cache_manager.get(plan, dataset_digest)
            assert cached_df is None
            assert cached_evidence is None

    def test_cache_clear(self):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            plan = Plan(filters=[Filter(column="test_col", op="=", value="test")])
            dataset_digest = "abc123"

            df = pl.DataFrame({"col1": [1, 2, 3]})
            evidence = {"test": "evidence"}

            # Store in cache
            cache_manager.put(plan, dataset_digest, df, evidence)

            # Should be available
            cached_df, cached_evidence = cache_manager.get(plan, dataset_digest)
            assert cached_df is not None

            # Clear cache
            cache_manager.clear()

            # Should no longer be available
            cached_df, cached_evidence = cache_manager.get(plan, dataset_digest)
            assert cached_df is None

    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            # Empty cache stats
            stats = cache_manager.stats()
            assert stats["total_files"] == 0
            assert stats["valid_entries"] == 0

            # Add some entries
            plan1 = Plan(filters=[Filter(column="col1", op="=", value="val1")])
            plan2 = Plan(filters=[Filter(column="col2", op="=", value="val2")])

            df1 = pl.DataFrame({"a": [1, 2]})
            df2 = pl.DataFrame({"b": [3, 4]})
            evidence = {"test": "evidence"}

            cache_manager.put(plan1, "digest1", df1, evidence)
            cache_manager.put(plan2, "digest2", df2, evidence)

            # Check stats
            stats = cache_manager.stats()
            assert stats["total_files"] == 4  # 2 parquet + 2 json
            assert stats["parquet_files"] == 2
            assert stats["json_files"] == 2
            assert stats["valid_entries"] == 2
            assert stats["ttl_hours"] == 24  # default

    def test_cache_invalid_files(self):
        """Test handling of invalid cache files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            plan = Plan(filters=[Filter(column="test_col", op="=", value="test")])
            dataset_digest = "abc123"

            # Create invalid cache files
            parquet_path, json_path = cache_manager._get_cache_paths(
                cache_manager._compute_fingerprint(plan, dataset_digest)
            )

            # Create empty files (invalid)
            parquet_path.touch()
            json_path.touch()

            # Should return None due to invalid files
            cached_df, cached_evidence = cache_manager.get(plan, dataset_digest)
            assert cached_df is None
            assert cached_evidence is None

    def test_cache_put_failure_handling(self):
        """Test that cache put failures are handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=Path(temp_dir) / "cache")

            plan = Plan(filters=[Filter(column="test_col", op="=", value="test")])
            dataset_digest = "abc123"

            # Create a DataFrame that might cause issues
            df = pl.DataFrame({"col1": [1, 2, 3]})
            evidence = {"test": "evidence"}

            # Mock a failure in the put method by patching the write_parquet method
            with patch.object(df, "write_parquet") as mock_write:
                mock_write.side_effect = Exception("Mocked failure")

                # Should not raise exception
                cache_manager.put(plan, dataset_digest, df, evidence)

                # Should not be in cache due to failure
                cached_df, cached_evidence = cache_manager.get(plan, dataset_digest)
                assert cached_df is None
                assert cached_evidence is None
