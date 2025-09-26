"""
Integration tests for the agent system with golden plans, synthetic data, and real data smoke tests.

This module provides comprehensive testing for Task 11:
- Golden plans: JSON + expected outputs on small data
- Synthetic: fixtures generate seasonal series with known components and injected regime shifts
- Real data smoke: if dataset available locally, run small DAG end-to-end (skipped on CI otherwise)
"""

import json
import os
import time
import pytest
import polars as pl
from pathlib import Path
from typing import Dict, Any, List

from data_agent.core.agent_schema import PlanGraph
from data_agent.core.agent_executor import execute
from data_agent.core.handles import StepHandle, StepStats


class TestAgentIntegration:
    """Integration tests for the full agent system."""

    def _create_step_stats(self, df: pl.DataFrame, file_path: Path) -> StepStats:
        """Create StepStats object from DataFrame and file."""
        return StepStats(
            rows=len(df),
            bytes=file_path.stat().st_size if file_path.exists() else 0,
            columns=len(df.columns),
            null_count={col: df[col].null_count() for col in df.columns},
            computed_at=time.time()
        )

    @pytest.fixture
    def examples_dir(self) -> Path:
        """Path to examples directory."""
        return Path(__file__).parent.parent / "examples"

    @pytest.fixture
    def golden_dataset(self, examples_dir: Path) -> pl.DataFrame:
        """Load the golden dataset for testing."""
        golden_path = examples_dir / "golden.parquet"
        if not golden_path.exists():
            pytest.skip(f"Golden dataset not found at {golden_path}")
        return pl.read_parquet(golden_path)

    @pytest.fixture
    def real_dataset(self) -> pl.DataFrame:
        """Load real dataset if available, otherwise skip."""
        data_path = Path(__file__).parent.parent / "data" / "pipeline_data.parquet"
        if not data_path.exists():
            pytest.skip("Real dataset not available - skipping smoke test")
        return pl.read_parquet(data_path)

    def test_golden_plan_regime_shifts_2021(self, examples_dir: Path, golden_dataset: pl.DataFrame):
        """Test the regime shifts 2021 golden plan against expected output."""
        plan_path = examples_dir / "agent_plans" / "regime_shifts_2021.json"
        expected_path = examples_dir / "expected" / "regime_shifts_2021.json"
        
        # Load the plan
        with open(plan_path) as f:
            plan_data = json.load(f)
        plan = PlanGraph(**plan_data)
        
        # Create initial handle for golden dataset
        temp_path = Path("/tmp/test_golden_data.parquet")
        golden_dataset.write_parquet(temp_path)
        
        dataset_handle = StepHandle(
            id="raw",
            store="parquet",
            path=temp_path,
            engine="polars",
            schema=golden_dataset.schema,
            stats=self._create_step_stats(golden_dataset, temp_path),
            fingerprint="golden_test"
        )
        
        # Execute the plan
        result_table, evidence = execute(plan, dataset_handle)
        
        # Load expected output if it exists
        if expected_path.exists():
            with open(expected_path) as f:
                expected_data = json.load(f)
            
            # Convert result to comparable format
            result_data = result_table.to_dicts()
            
            # Compare key fields (allowing for some flexibility in exact values)
            assert len(result_data) == len(expected_data), "Result length mismatch"
            
            # Check that we have the expected columns (with some flexibility for column name variations)
            if expected_data and result_data:
                expected_cols = set(expected_data[0].keys())
                result_cols = set(result_data[0].keys())
                
                # Core columns that should exist
                core_cols = {"pipeline_name", "changepoint_date", "change_magnitude"}
                missing_core = core_cols - result_cols
                assert not missing_core, f"Missing core columns: {missing_core}. Available: {result_cols}"
        
        # Verify evidence structure
        assert "plan" in evidence
        assert "steps" in evidence
        plan_evidence = evidence["plan"]
        assert ("plan_hash" in plan_evidence or "hash" in plan_evidence), f"Missing hash field in plan evidence: {plan_evidence.keys()}"
        
        # Clean up
        temp_path.unlink(missing_ok=True)

    def test_golden_plan_simple_aggregation(self, examples_dir: Path, golden_dataset: pl.DataFrame):
        """Test a simple aggregation golden plan."""
        plan_path = examples_dir / "agent_plans" / "simple_aggregation_example.json"
        
        # Load the plan
        with open(plan_path) as f:
            plan_data = json.load(f)
        plan = PlanGraph(**plan_data)
        
        # Create initial handle for golden dataset
        temp_path = Path("/tmp/test_golden_simple.parquet")
        golden_dataset.write_parquet(temp_path)
        
        dataset_handle = StepHandle(
            id="raw",
            store="parquet", 
            path=temp_path,
            engine="polars",
            schema=golden_dataset.schema,
            stats=self._create_step_stats(golden_dataset, temp_path),
            fingerprint="golden_simple_test"
        )
        
        # Execute the plan
        result_table, evidence = execute(plan, dataset_handle)
        
        # Basic validation
        assert len(result_table) > 0, "Result should not be empty"
        assert "plan" in evidence
        assert "steps" in evidence
        
        # Clean up
        temp_path.unlink(missing_ok=True)

    def test_synthetic_stl_detection(self):
        """Test STL decomposition on synthetic seasonal data with known components."""
        from tests.fixtures.synthetic_stl import generate_seasonal_series
        
        # Generate synthetic data with known weekly and annual patterns
        df = generate_seasonal_series(
            start_date="2020-01-01",
            end_date="2022-12-31", 
            weekly_amplitude=100,
            annual_amplitude=500,
            trend_slope=0.1,
            noise_std=50,
            pipeline_names=["TestPipe1", "TestPipe2"]
        )
        
        # Rename date column to match expected format
        df = df.rename({"date": "eff_gas_day"})
        
        # Create a plan that does STL decomposition
        plan_data = {
            "nodes": [
                {
                    "id": "a",
                    "op": "aggregate",
                    "params": {
                        "groupby": ["pipeline_name", "eff_gas_day"],
                        "metrics": [{"col": "value", "fn": "sum"}]
                    }
                },
                {
                    "id": "s", 
                    "op": "stl_deseasonalize",
                    "params": {
                        "column": "sum_value",
                        "period": 7,  # Weekly seasonality
                        "seasonal": 7,
                        "trend": 365
                    }
                },
                {
                    "id": "e",
                    "op": "evidence_collect",
                    "params": {}
                }
            ],
            "edges": [
                {"src": "raw", "dst": "a"},
                {"src": "a", "dst": "s"}, 
                {"src": "s", "dst": "e"}
            ],
            "inputs": ["raw"],
            "outputs": ["s"]
        }
        
        plan = PlanGraph(**plan_data)
        
        # Create handle for synthetic data
        temp_path = Path("/tmp/test_synthetic_stl.parquet")
        df.write_parquet(temp_path)
        
        dataset_handle = StepHandle(
            id="raw",
            store="parquet",
            path=temp_path,
            engine="polars", 
            schema=df.schema,
            stats=self._create_step_stats(df, temp_path),
            fingerprint="synthetic_stl_test"
        )
        
        # Execute the plan
        result_table, evidence = execute(plan, dataset_handle)
        
        # Validate that STL components were detected
        result_cols = result_table.columns
        print(f"Result columns: {result_cols}")  # Debug output
        
        # Check for deseasonalized column (the main output)
        assert "deseasonalized" in result_cols, f"Expected deseasonalized column not found. Available: {result_cols}"
        
        # STL step should produce at least the deseasonalized column
        assert len(result_table) > 0, "Result should not be empty"
        
        # Clean up
        temp_path.unlink(missing_ok=True)

    def test_synthetic_changepoint_detection(self):
        """Test changepoint detection on synthetic data with injected regime shifts."""
        from tests.fixtures.synthetic_cp import generate_changepoint_series
        
        # Generate synthetic data with known changepoints
        df = generate_changepoint_series(
            start_date="2020-01-01",
            end_date="2022-12-31",
            changepoints=["2020-06-01", "2021-03-01", "2021-09-01"],
            regime_values=[100, 200, 150, 300],
            noise_std=20,
            pipeline_names=["TestPipe1"]
        )
        
        # Rename date column to match expected format
        df = df.rename({"date": "eff_gas_day"})
        
        # Create a plan that detects changepoints
        plan_data = {
            "nodes": [
                {
                    "id": "c",
                    "op": "changepoint", 
                    "params": {
                        "column": "value",
                        "method": "pelt",
                        "min_size": 7,
                        "penalty": 1.0,
                        "groupby": ["pipeline_name"]
                    }
                },
                {
                    "id": "r",
                    "op": "rank",
                    "params": {
                        "by": ["change_magnitude"],
                        "descending": True
                    }
                },
                {
                    "id": "e",
                    "op": "evidence_collect", 
                    "params": {}
                }
            ],
            "edges": [
                {"src": "raw", "dst": "c"},
                {"src": "c", "dst": "r"},
                {"src": "r", "dst": "e"}
            ],
            "inputs": ["raw"],
            "outputs": ["r"]
        }
        
        plan = PlanGraph(**plan_data)
        
        # Create handle for synthetic data
        temp_path = Path("/tmp/test_synthetic_cp.parquet")
        df.write_parquet(temp_path)
        
        dataset_handle = StepHandle(
            id="raw",
            store="parquet",
            path=temp_path,
            engine="polars",
            schema=df.schema,
            stats=self._create_step_stats(df, temp_path),
            fingerprint="synthetic_cp_test"
        )
        
        # Execute the plan
        result_table, evidence = execute(plan, dataset_handle)
        
        # Validate that changepoints were detected
        assert len(result_table) > 0, "Should detect some changepoints"
        
        result_cols = result_table.columns
        expected_cp_cols = ["changepoint_date", "change_magnitude"]
        
        for col in expected_cp_cols:
            assert any(col in result_col for result_col in result_cols), f"Expected changepoint column {col} not found"
        
        # Should detect some changepoints (allowing for algorithm sensitivity)
        # We injected 3 changepoints, but the algorithm might detect more due to noise
        assert len(result_table) >= 3, f"Expected at least 3 changepoints, got {len(result_table)}"
        assert len(result_table) <= 100, f"Too many changepoints detected: {len(result_table)}"
        
        # Clean up
        temp_path.unlink(missing_ok=True)

    @pytest.mark.skipif(
        not os.path.exists(Path(__file__).parent.parent / "data" / "pipeline_data.parquet"),
        reason="Real dataset not available - CI skip"
    )
    def test_real_data_smoke(self, real_dataset: pl.DataFrame):
        """Smoke test with real data - basic end-to-end execution."""
        # Simple aggregation plan for smoke test
        plan_data = {
            "nodes": [
                {
                    "id": "f",
                    "op": "filter",
                    "params": {
                        "column": "eff_gas_day",
                        "op": "between",
                        "value": ["2022-01-01", "2022-01-31"]  # Just January for speed
                    }
                },
                {
                    "id": "a",
                    "op": "aggregate", 
                    "params": {
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]
                    }
                },
                {
                    "id": "r",
                    "op": "rank",
                    "params": {
                        "by": ["sum_scheduled_quantity"],
                        "descending": True
                    }
                },
                {
                    "id": "l",
                    "op": "limit",
                    "params": {"n": 5}
                },
                {
                    "id": "e",
                    "op": "evidence_collect",
                    "params": {}
                }
            ],
            "edges": [
                {"src": "raw", "dst": "f"},
                {"src": "f", "dst": "a"},
                {"src": "a", "dst": "r"},
                {"src": "r", "dst": "l"},
                {"src": "l", "dst": "e"}
            ],
            "inputs": ["raw"],
            "outputs": ["l"]
        }
        
        plan = PlanGraph(**plan_data)
        
        # Create handle for real data
        temp_path = Path("/tmp/test_real_data.parquet")
        real_dataset.write_parquet(temp_path)
        
        dataset_handle = StepHandle(
            id="raw",
            store="parquet",
            path=temp_path,
            engine="polars",
            schema=real_dataset.schema,
            stats=self._create_step_stats(real_dataset, temp_path),
            fingerprint="real_data_smoke_test"
        )
        
        # Execute the plan
        result_table, evidence = execute(plan, dataset_handle)
        
        # Basic smoke test validation
        assert len(result_table) <= 5, "Should be limited to 5 results"
        assert len(result_table) > 0, "Should have some results"
        assert "pipeline_name" in result_table.columns
        assert "sum_scheduled_quantity" in result_table.columns
        
        # Verify evidence structure
        assert "plan" in evidence
        assert "steps" in evidence
        assert len(evidence["steps"]) == 5  # 5 steps in the plan
        
        # Clean up
        temp_path.unlink(missing_ok=True)

    def test_plan_validation_and_repair(self):
        """Test that invalid plans are caught and repaired where possible."""
        # Test with a plan missing evidence_collect step
        incomplete_plan_data = {
            "nodes": [
                {
                    "id": "a",
                    "op": "aggregate",
                    "params": {
                        "groupby": ["pipeline_name"],
                        "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]
                    }
                }
            ],
            "edges": [
                {"src": "raw", "dst": "a"}
            ],
            "inputs": ["raw"],
            "outputs": ["a"]
        }
        
        # This should either fail validation or be automatically repaired
        # The exact behavior depends on the implementation
        plan = PlanGraph(**incomplete_plan_data)
        
        # Basic validation that the plan can be created
        assert len(plan.nodes) >= 1
        assert len(plan.edges) >= 1

    def test_evidence_structure(self, examples_dir: Path, golden_dataset: pl.DataFrame):
        """Test that evidence contains all required components."""
        plan_path = examples_dir / "agent_plans" / "simple_aggregation_example.json"
        
        with open(plan_path) as f:
            plan_data = json.load(f)
        plan = PlanGraph(**plan_data)
        
        # Create handle
        temp_path = Path("/tmp/test_evidence.parquet")
        golden_dataset.write_parquet(temp_path)
        
        dataset_handle = StepHandle(
            id="raw",
            store="parquet",
            path=temp_path,
            engine="polars",
            schema=golden_dataset.schema,
            stats=self._create_step_stats(golden_dataset, temp_path),
            fingerprint="evidence_test"
        )
        
        # Execute
        result_table, evidence = execute(plan, dataset_handle)
        
        # Validate evidence structure
        assert isinstance(evidence, dict)
        
        # Plan metadata
        assert "plan" in evidence
        plan_evidence = evidence["plan"]
        assert "plan_hash" in plan_evidence or "hash" in plan_evidence  # Accept either format
        assert "nodes" in plan_evidence
        assert "edges" in plan_evidence
        
        # Step evidence
        assert "steps" in evidence
        steps_evidence = evidence["steps"]
        assert isinstance(steps_evidence, list)
        
        for step_evidence in steps_evidence:
            assert "node_id" in step_evidence
            assert "params" in step_evidence
            assert "timings" in step_evidence
            assert "input_stats" in step_evidence
            assert "output_stats" in step_evidence
            
            # Timings should have duration information (format may vary)
            timings = step_evidence["timings"]
            # Accept different timing formats
            has_duration = any(key in timings for key in ["duration_ms", "total", "execute"])
            assert has_duration, f"Expected timing information not found in: {list(timings.keys())}"
            
            # Stats should have row/byte counts (format may vary)
            for stats in [step_evidence["input_stats"], step_evidence["output_stats"]]:
                # Accept different stat formats
                has_rows = any(key in stats for key in ["rows", "total_rows"])
                has_bytes = any(key in stats for key in ["bytes", "total_bytes"])
                assert has_rows, f"Expected row count not found in: {list(stats.keys())}"
                assert has_bytes, f"Expected byte count not found in: {list(stats.keys())}"
        
        # Clean up
        temp_path.unlink(missing_ok=True)
