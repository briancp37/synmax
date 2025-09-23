"""Cache management for query results."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from ..config import CACHE_DIR, DATA_AGENT_CACHE_TTL_HOURS
from ..core.plan_schema import Plan


class CacheManager:
    """Manages caching of query results with fingerprinting and TTL."""

    def __init__(self, cache_dir: Path | None = None, ttl_hours: int | None = None):
        """Initialize cache manager.

        Args:
            cache_dir: Cache directory path (defaults to config)
            ttl_hours: Time-to-live in hours (defaults to config)
        """
        self.cache_dir = cache_dir or CACHE_DIR / "results"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours or DATA_AGENT_CACHE_TTL_HOURS
        self.ttl_seconds = self.ttl_hours * 3600

    def _compute_fingerprint(self, plan: Plan, dataset_digest: str) -> str:
        """Compute fingerprint from canonical Plan JSON + dataset digest.

        Args:
            plan: Query plan
            dataset_digest: Dataset digest string

        Returns:
            Hexadecimal fingerprint string
        """
        # Create canonical JSON representation of plan
        # Use default=str to handle Polars expressions that can't be serialized
        plan_dict = plan.model_dump()
        plan_json = orjson.dumps(plan_dict, option=orjson.OPT_SORT_KEYS, default=str).decode(
            "utf-8"
        )

        # Combine with dataset digest
        combined = f"{plan_json}|{dataset_digest}".encode()

        # Compute SHA-256 hash
        return hashlib.sha256(combined).hexdigest()

    def _get_cache_paths(self, fingerprint: str) -> tuple[Path, Path]:
        """Get cache file paths for a fingerprint.

        Args:
            fingerprint: Cache fingerprint

        Returns:
            Tuple of (parquet_path, json_path)
        """
        parquet_path = self.cache_dir / f"{fingerprint}.parquet"
        json_path = self.cache_dir / f"{fingerprint}.json"
        return parquet_path, json_path

    def _is_cache_valid(self, parquet_path: Path, json_path: Path) -> bool:
        """Check if cache files exist and are within TTL.

        Args:
            parquet_path: Path to parquet cache file
            json_path: Path to JSON cache file

        Returns:
            True if cache is valid, False otherwise
        """
        if not parquet_path.exists() or not json_path.exists():
            return False

        # Check TTL based on file modification time
        current_time = time.time()
        file_mtime = parquet_path.stat().st_mtime

        return (current_time - file_mtime) < self.ttl_seconds

    def get(
        self, plan: Plan, dataset_digest: str
    ) -> tuple[pl.DataFrame | None, dict[str, Any] | None]:
        """Retrieve cached result if available and valid.

        Args:
            plan: Query plan
            dataset_digest: Dataset digest string

        Returns:
            Tuple of (DataFrame, evidence_dict) or (None, None) if not cached
        """
        fingerprint = self._compute_fingerprint(plan, dataset_digest)
        parquet_path, json_path = self._get_cache_paths(fingerprint)

        if not self._is_cache_valid(parquet_path, json_path):
            return None, None

        try:
            # Load cached DataFrame
            df = pl.read_parquet(parquet_path)

            # Load cached evidence
            with open(json_path, "rb") as f:
                evidence = orjson.loads(f.read())

            # Verify that the evidence contains the correct fingerprint
            if evidence.get("cache", {}).get("fingerprint") != fingerprint:
                return None, None

            return df, evidence
        except Exception:
            # If loading fails, return None (cache miss)
            return None, None

    def put(
        self, plan: Plan, dataset_digest: str, df: pl.DataFrame, evidence: dict[str, Any]
    ) -> None:
        """Store result in cache.

        Args:
            plan: Query plan
            dataset_digest: Dataset digest string
            df: Result DataFrame
            evidence: Evidence dictionary
        """
        fingerprint = self._compute_fingerprint(plan, dataset_digest)
        parquet_path, json_path = self._get_cache_paths(fingerprint)

        try:
            # Store DataFrame
            df.write_parquet(parquet_path)

            # Store evidence with cache metadata
            cache_evidence = evidence.copy()
            cache_evidence["cache"] = {
                "hit": False,  # This is a cache put, not a hit
                "fingerprint": fingerprint,
                "ttl_hours": self.ttl_hours,
                "cached_at": time.time(),
            }

            with open(json_path, "wb") as f:
                f.write(orjson.dumps(cache_evidence, option=orjson.OPT_INDENT_2))

        except Exception:
            # If storing fails, silently continue (non-critical)
            pass

    def clear(self) -> None:
        """Clear all cached results."""
        if self.cache_dir.exists():
            for file_path in self.cache_dir.glob("*"):
                file_path.unlink()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size_bytes": 0, "valid_entries": 0}

        files = list(self.cache_dir.glob("*"))
        parquet_files = [f for f in files if f.suffix == ".parquet"]
        json_files = [f for f in files if f.suffix == ".json"]

        total_size = sum(f.stat().st_size for f in files)

        # Count valid entries (both parquet and json exist and are within TTL)
        valid_entries = 0
        for parquet_file in parquet_files:
            json_file = parquet_file.with_suffix(".json")
            if self._is_cache_valid(parquet_file, json_file):
                valid_entries += 1

        return {
            "total_files": len(files),
            "parquet_files": len(parquet_files),
            "json_files": len(json_files),
            "total_size_bytes": total_size,
            "valid_entries": valid_entries,
            "ttl_hours": self.ttl_hours,
        }
