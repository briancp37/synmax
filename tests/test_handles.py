"""Tests for step handles, storage, and hybrid materialization."""

import tempfile
import time
from pathlib import Path

import polars as pl
import pytest

from data_agent.core.handles import (
    HandleStorage,
    StepHandle,
    StepStats,
    create_lazy_handle,
    create_memory_handle,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield HandleStorage(Path(temp_dir))


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "category": ["A", "B", "A", "C", "B"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        }
    ).with_columns(pl.col("date").str.to_date())


class TestStepHandle:
    """Test StepHandle data class."""

    def test_memory_handle_creation(self):
        """Test creating memory handles."""
        schema = {"id": "Int64", "value": "Float64"}
        stats = StepStats(
            rows=100,
            bytes=1024,
            columns=2,
            null_count={"id": 0, "value": 5},
            computed_at=time.time(),
        )

        handle = create_memory_handle("test_step", schema=schema, stats=stats)

        assert handle.id == "test_step"
        assert handle.store == "memory"
        assert handle.path is None
        assert handle.schema == schema
        assert handle.stats == stats
        assert handle.fingerprint is None
        assert handle.engine == "polars"

    def test_lazy_handle_creation(self):
        """Test creating lazy handles."""
        schema = {"id": "Int64", "value": "Float64"}

        handle = create_lazy_handle("test_step", schema=schema, engine="duckdb")

        assert handle.id == "test_step"
        assert handle.store == "lazy"
        assert handle.path is None
        assert handle.schema == schema
        assert handle.stats is None
        assert handle.fingerprint is None
        assert handle.engine == "duckdb"

    def test_handle_validation(self):
        """Test handle validation in __post_init__."""
        # Should raise error for parquet store without path
        with pytest.raises(ValueError, match="Parquet store requires a path"):
            StepHandle(id="test", store="parquet", path=None)

        # Should raise error for memory store with path
        with pytest.raises(ValueError, match="Memory store should not have a path"):
            StepHandle(id="test", store="memory", path=Path("/tmp/test.parquet"))


class TestHandleStorage:
    """Test HandleStorage class."""

    def test_fingerprint_computation(self, temp_storage):
        """Test fingerprint computation is deterministic and unique."""
        params1 = {"filter": "value > 10", "columns": ["id", "value"]}
        params2 = {"columns": ["id", "value"], "filter": "value > 10"}  # Same but different order
        params3 = {"filter": "value > 20", "columns": ["id", "value"]}  # Different filter

        input_fps = ["abc123", "def456"]
        dataset_digest = "dataset_hash_123"

        # Same params (different order) should produce same fingerprint
        fp1 = temp_storage.compute_fingerprint("step1", params1, input_fps, dataset_digest)
        fp2 = temp_storage.compute_fingerprint("step1", params2, input_fps, dataset_digest)
        assert fp1 == fp2

        # Different params should produce different fingerprint
        fp3 = temp_storage.compute_fingerprint("step1", params3, input_fps, dataset_digest)
        assert fp1 != fp3

        # Different step ID should produce different fingerprint
        fp4 = temp_storage.compute_fingerprint("step2", params1, input_fps, dataset_digest)
        assert fp1 != fp4

        # Different input fingerprints should produce different fingerprint
        fp5 = temp_storage.compute_fingerprint("step1", params1, ["xyz789"], dataset_digest)
        assert fp1 != fp5

    def test_storage_path_sharding(self, temp_storage):
        """Test storage path uses proper 2-level sharding."""
        fingerprint = "abcd1234567890"
        path = temp_storage.get_storage_path(fingerprint)

        # Should be: base_dir/ab/cd/abcd1234567890/part-000.parquet
        expected_parts = [temp_storage.base_dir, "ab", "cd", fingerprint, "part-000.parquet"]
        expected_path = Path(*expected_parts)

        assert path == expected_path
        assert path.parent.exists()  # Directory should be created

    def test_checkpoint_heuristics(self, temp_storage):
        """Test checkpoint decision heuristics."""
        # Test expensive operations
        assert temp_storage.should_checkpoint(None, "stl_deseasonalize", False)
        assert temp_storage.should_checkpoint(None, "changepoint", False)
        assert temp_storage.should_checkpoint(None, "join", False)

        # Test multiple consumers
        assert temp_storage.should_checkpoint(None, "filter", True)

        # Test size thresholds
        large_stats = StepStats(
            rows=200_000, bytes=200_000_000, columns=10, null_count={}, computed_at=time.time()
        )
        assert temp_storage.should_checkpoint(large_stats, "filter", False)

        # Test small operation should not checkpoint
        small_stats = StepStats(
            rows=1000, bytes=10_000, columns=5, null_count={}, computed_at=time.time()
        )
        assert not temp_storage.should_checkpoint(small_stats, "filter", False)

    def test_materialize_and_load_handle(self, temp_storage, sample_df):
        """Test materializing DataFrame and loading from handle."""
        fingerprint = "test_fingerprint_123"
        step_id = "test_step"

        # Materialize the DataFrame
        handle = temp_storage.materialize_handle(sample_df, fingerprint, step_id)

        # Verify handle properties
        assert handle.id == step_id
        assert handle.store == "parquet"
        assert handle.path is not None
        assert handle.path.exists()
        assert handle.fingerprint == fingerprint
        assert handle.stats is not None
        assert handle.stats.rows == 5
        assert handle.stats.columns == 4
        assert handle.schema is not None

        # Verify we can load the data back
        lf = temp_storage.load_from_handle(handle)
        loaded_df = lf.collect()

        # Data should match (may be different column order)
        assert loaded_df.height == sample_df.height
        assert set(loaded_df.columns) == set(sample_df.columns)

        # Sort both for comparison
        original_sorted = sample_df.sort("id")
        loaded_sorted = loaded_df.sort("id")
        assert original_sorted.equals(loaded_sorted)

    def test_load_from_lazy_handle_fails(self, temp_storage):
        """Test that loading from lazy handle fails appropriately."""
        handle = create_lazy_handle("test_step")

        with pytest.raises(ValueError, match="Cannot load from lazy handle"):
            temp_storage.load_from_handle(handle)

    def test_cleanup_expired(self, temp_storage, sample_df):
        """Test cleanup of expired files."""
        # Create some test files with different ages
        old_fingerprint = "old_file_123"
        new_fingerprint = "new_file_456"

        # Create old handle (we'll manually set its mtime)
        old_handle = temp_storage.materialize_handle(sample_df, old_fingerprint, "old_step")

        # Create new handle
        new_handle = temp_storage.materialize_handle(sample_df, new_fingerprint, "new_step")

        # Manually set old file's modification time to be expired
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        import os

        os.utime(old_handle.path, (old_time, old_time))

        # Run cleanup with 24-hour TTL
        cleaned_count = temp_storage.cleanup_expired(ttl_hours=24)

        assert cleaned_count == 1  # Should clean up old file
        assert not old_handle.path.exists()  # Old file should be gone
        assert new_handle.path.exists()  # New file should remain

    def test_storage_stats(self, temp_storage, sample_df):
        """Test storage statistics."""
        # Initially empty
        stats = temp_storage.get_storage_stats()
        assert stats["total_files"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["total_size_mb"] == 0.0

        # Add some files
        temp_storage.materialize_handle(sample_df, "fp1", "step1")
        temp_storage.materialize_handle(sample_df, "fp2", "step2")

        stats = temp_storage.get_storage_stats()
        assert stats["total_files"] == 2
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_mb"] > 0
        assert stats["directories"] >= 2  # At least 2 hash directories


class TestIntegration:
    """Integration tests for the handles system."""

    def test_full_workflow(self, temp_storage, sample_df):
        """Test complete workflow from fingerprint to materialization to cleanup."""
        step_id = "integration_test"
        params = {"operation": "filter", "condition": "value > 25"}
        input_fingerprints = ["input_123"]
        dataset_digest = "dataset_abc"

        # 1. Compute fingerprint
        fingerprint = temp_storage.compute_fingerprint(
            step_id, params, input_fingerprints, dataset_digest
        )
        assert len(fingerprint) == 64  # SHA-256 hex length

        # 2. Check if we should checkpoint (simulate large operation)
        stats = StepStats(
            rows=sample_df.height,
            bytes=1000000,  # Simulate large size
            columns=sample_df.width,
            null_count={col: sample_df[col].null_count() for col in sample_df.columns},
            computed_at=time.time(),
        )
        should_checkpoint = temp_storage.should_checkpoint(stats, "changepoint", False)
        assert should_checkpoint  # changepoint is expensive

        # 3. Materialize if needed
        if should_checkpoint:
            handle = temp_storage.materialize_handle(sample_df, fingerprint, step_id)
            assert handle.store == "parquet"
            assert handle.path.exists()

        # 4. Load back and verify
        lf = temp_storage.load_from_handle(handle)
        result_df = lf.collect()
        assert result_df.equals(sample_df.sort("id"))

        # 5. Check storage stats
        stats = temp_storage.get_storage_stats()
        assert stats["total_files"] == 1
        assert stats["total_size_bytes"] > 0

        # 6. Cleanup (shouldn't remove fresh files)
        cleaned = temp_storage.cleanup_expired(ttl_hours=1)
        assert cleaned == 0
        assert handle.path.exists()

        # 7. Force cleanup by using very short TTL
        cleaned = temp_storage.cleanup_expired(ttl_hours=0)
        assert cleaned == 1
        assert not handle.path.exists()

    def test_fingerprint_stability_across_runs(self, temp_storage):
        """Test that fingerprints are stable across different runs."""
        params = {"filter": "value > 10", "sort": ["id", "date"]}
        inputs = ["input1", "input2"]
        dataset = "dataset_hash"
        step_id = "stable_test"

        # Compute fingerprint multiple times
        fingerprints = []
        for _ in range(5):
            fp = temp_storage.compute_fingerprint(step_id, params, inputs, dataset)
            fingerprints.append(fp)

        # All should be identical
        assert len(set(fingerprints)) == 1

        # Small change should produce different fingerprint
        modified_params = params.copy()
        modified_params["filter"] = "value > 11"
        different_fp = temp_storage.compute_fingerprint(step_id, modified_params, inputs, dataset)
        assert different_fp != fingerprints[0]


@pytest.mark.parametrize("engine", ["polars", "duckdb"])
def test_handle_with_different_engines(engine):
    """Test handles work with different engines."""
    handle = create_memory_handle("test", engine=engine)
    assert handle.engine == engine

    lazy_handle = create_lazy_handle("test", engine=engine)
    assert lazy_handle.engine == engine


def test_step_stats():
    """Test StepStats dataclass."""
    stats = StepStats(
        rows=1000,
        bytes=50000,
        columns=5,
        null_count={"col1": 10, "col2": 0},
        computed_at=time.time(),
    )

    assert stats.rows == 1000
    assert stats.bytes == 50000
    assert stats.columns == 5
    assert stats.null_count == {"col1": 10, "col2": 0}
    assert isinstance(stats.computed_at, float)
