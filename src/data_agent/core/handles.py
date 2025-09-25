"""Step handles for hybrid materialization and content-addressed storage."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from ..config import ARTIFACTS_DIR


@dataclass
class StepStats:
    """Statistics for a step's data."""

    rows: int
    bytes: int
    columns: int
    null_count: dict[str, int]
    computed_at: float  # timestamp


@dataclass
class StepHandle:
    """Handle to a materialized step result with content addressing."""

    id: str  # Step node ID
    store: Literal["memory", "parquet", "lazy"] = "lazy"
    path: Path | None = None  # Path to materialized parquet file
    engine: Literal["polars", "duckdb"] = "polars"
    schema: dict[str, str] | None = None  # Column name -> dtype mapping
    stats: StepStats | None = None
    fingerprint: str | None = None  # Content hash for cache addressing

    def __post_init__(self) -> None:
        """Validate handle after creation."""
        if self.store == "parquet" and self.path is None:
            raise ValueError("Parquet store requires a path")
        if self.store == "memory" and self.path is not None:
            raise ValueError("Memory store should not have a path")


class HandleStorage:
    """Manages content-addressed storage for step handles."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize handle storage.

        Args:
            base_dir: Base directory for storage (defaults to artifacts/tmp)
        """
        self.base_dir = base_dir or (ARTIFACTS_DIR / "tmp")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def compute_fingerprint(
        self,
        step_id: str,
        params: dict[str, Any],
        input_fingerprints: list[str],
        dataset_digest: str,
    ) -> str:
        """Compute content-addressed fingerprint for a step.

        Args:
            step_id: Step identifier
            params: Step parameters (canonicalized)
            input_fingerprints: Fingerprints of input handles
            dataset_digest: Base dataset digest

        Returns:
            SHA-256 fingerprint as hex string
        """
        # Create canonical representation
        canonical_inputs = "|".join(sorted(input_fingerprints))
        canonical_params = str(sorted(params.items()))

        combined = f"{step_id}|{canonical_params}|{canonical_inputs}|{dataset_digest}"

        return hashlib.sha256(combined.encode()).hexdigest()

    def get_storage_path(self, fingerprint: str) -> Path:
        """Get storage path using 2-level directory sharding.

        Layout: artifacts/tmp/<2>/<2>/<hash>/part-000.parquet

        Args:
            fingerprint: Content fingerprint

        Returns:
            Path to parquet file
        """
        # Use first 4 characters for 2-level sharding
        level1 = fingerprint[:2]
        level2 = fingerprint[2:4]

        dir_path = self.base_dir / level1 / level2 / fingerprint
        dir_path.mkdir(parents=True, exist_ok=True)

        return dir_path / "part-000.parquet"

    def should_checkpoint(
        self,
        stats: StepStats | None,
        op_type: str,
        has_multiple_consumers: bool,
        row_threshold: int = 100_000,
        byte_threshold: int = 100_000_000,
    ) -> bool:
        """Determine if a step should be checkpointed.

        Args:
            stats: Step statistics (None if not available)
            op_type: Operation type
            has_multiple_consumers: Whether step has multiple downstream consumers
            row_threshold: Row count threshold for checkpointing
            byte_threshold: Byte size threshold for checkpointing

        Returns:
            True if step should be checkpointed
        """
        # Always checkpoint expensive operations
        expensive_ops = {"stl_deseasonalize", "changepoint", "join"}
        if op_type in expensive_ops:
            return True

        # Always checkpoint if multiple consumers
        if has_multiple_consumers:
            return True

        # Checkpoint if exceeds size thresholds
        if stats is not None:
            if stats.rows > row_threshold or stats.bytes > byte_threshold:
                return True

        return False

    def materialize_handle(
        self, df: pl.DataFrame, fingerprint: str, step_id: str, engine: str = "polars"
    ) -> StepHandle:
        """Materialize a DataFrame to disk and create handle.

        Args:
            df: DataFrame to materialize
            fingerprint: Content fingerprint
            step_id: Step identifier
            engine: Execution engine used

        Returns:
            StepHandle pointing to materialized data
        """
        storage_path = self.get_storage_path(fingerprint)

        # Write with compression and optimal row group size
        df.write_parquet(
            storage_path, compression="zstd", row_group_size=128 * 1024 * 1024  # 128MB row groups
        )

        # Compute stats
        stats = StepStats(
            rows=df.height,
            bytes=storage_path.stat().st_size,
            columns=df.width,
            null_count={col: df[col].null_count() for col in df.columns},
            computed_at=time.time(),
        )

        # Create schema mapping
        schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

        return StepHandle(
            id=step_id,
            store="parquet",
            path=storage_path,
            engine=engine,
            schema=schema,
            stats=stats,
            fingerprint=fingerprint,
        )

    def load_from_handle(self, handle: StepHandle) -> pl.LazyFrame:
        """Load data from a step handle as lazy frame.

        Args:
            handle: Step handle to load from

        Returns:
            Lazy frame for the handle's data
        """
        if handle.store == "parquet" and handle.path is not None:
            return pl.scan_parquet(handle.path)
        elif handle.store == "lazy":
            # For lazy handles, this should not be called directly
            # The caller should maintain the lazy frame reference
            raise ValueError("Cannot load from lazy handle without LazyFrame reference")
        else:
            raise ValueError(f"Cannot load from handle with store type: {handle.store}")

    def cleanup_expired(self, ttl_hours: int = 24) -> int:
        """Clean up expired checkpoint files.

        Args:
            ttl_hours: Time-to-live in hours

        Returns:
            Number of files cleaned up
        """
        if not self.base_dir.exists():
            return 0

        ttl_seconds = ttl_hours * 3600
        current_time = time.time()
        cleaned_count = 0

        # Walk through all parquet files in storage
        try:
            parquet_files = list(self.base_dir.rglob("*.parquet"))
        except (OSError, FileNotFoundError):
            # Directory structure might be corrupted, return early
            return 0

        for parquet_file in parquet_files:
            try:
                file_age = current_time - parquet_file.stat().st_mtime

                if file_age > ttl_seconds:
                    parquet_file.unlink()
                    # Also remove parent directory if empty
                    parent = parquet_file.parent
                    if parent != self.base_dir and parent.exists():
                        try:
                            # Check if directory is empty before removing
                            if not any(parent.iterdir()):
                                parent.rmdir()
                                # Also try to remove grandparent if empty
                                grandparent = parent.parent
                                if grandparent != self.base_dir and grandparent.exists():
                                    if not any(grandparent.iterdir()):
                                        grandparent.rmdir()
                        except OSError:
                            # Directory not empty or permission issue, skip
                            pass
                    cleaned_count += 1
            except (OSError, FileNotFoundError):
                # File might be in use or already removed, skip
                pass

        return cleaned_count

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        if not self.base_dir.exists():
            return {"total_files": 0, "total_size_bytes": 0, "total_size_mb": 0.0, "directories": 0}

        parquet_files = list(self.base_dir.rglob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)
        directories = len(list(self.base_dir.rglob("*/")))

        return {
            "total_files": len(parquet_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "directories": directories,
            "base_dir": str(self.base_dir),
        }


def create_memory_handle(
    step_id: str,
    schema: dict[str, str] | None = None,
    stats: StepStats | None = None,
    engine: str = "polars",
) -> StepHandle:
    """Create a handle for in-memory data.

    Args:
        step_id: Step identifier
        schema: Column schema mapping
        stats: Step statistics
        engine: Execution engine

    Returns:
        Memory-based step handle
    """
    return StepHandle(
        id=step_id,
        store="memory",
        path=None,
        engine=engine,
        schema=schema,
        stats=stats,
        fingerprint=None,
    )


def create_lazy_handle(
    step_id: str, schema: dict[str, str] | None = None, engine: str = "polars"
) -> StepHandle:
    """Create a handle for lazy computation.

    Args:
        step_id: Step identifier
        schema: Column schema mapping (if known)
        engine: Execution engine

    Returns:
        Lazy computation handle
    """
    return StepHandle(
        id=step_id,
        store="lazy",
        path=None,
        engine=engine,
        schema=schema,
        stats=None,
        fingerprint=None,
    )
