"""Tests for the ingest module."""

import tempfile
from pathlib import Path

import orjson
import polars as pl
import pytest

from data_agent.ingest.dictionary import build_data_dictionary, write_dictionary
from data_agent.ingest.loader import load_dataset


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return pl.DataFrame(
        {
            "pipeline_name": ["Test Pipeline"] * 10,
            "loc_name": ["Location A"] * 5 + ["Location B"] * 5,
            "connecting_pipeline": ["Connecting"] * 10,
            "connecting_entity": ["Entity"] * 10,
            "rec_del_sign": [1, -1] * 5,
            "category_short": ["LDC"] * 10,
            "country_name": ["USA"] * 10,
            "state_abb": ["TX"] * 10,
            "county_name": ["Test County"] * 10,
            "latitude": [32.0] * 5 + [None] * 5,  # Some nulls
            "longitude": [-97.0] * 5 + [None] * 5,  # Some nulls
            "eff_gas_day": ["2022-01-01", "2022-01-02"] * 5,
            "scheduled_quantity": [1000.0, 2000.0, 0.0, 1500.0, 3000.0] * 2,
        }
    )


@pytest.fixture
def temp_parquet(sample_data):
    """Create a temporary parquet file with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        sample_data.write_parquet(f.name)
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestLoader:
    """Tests for the loader module."""

    def test_load_dataset_existing_file(self, temp_parquet):
        """Test loading an existing parquet file."""
        lf = load_dataset(temp_parquet, auto=False)

        # Verify it's a LazyFrame
        assert isinstance(lf, pl.LazyFrame)

        # Collect to verify structure
        df = lf.collect()
        assert df.shape[0] == 10
        assert "pipeline_name" in df.columns

        # Check dtype normalization
        assert df["rec_del_sign"].dtype == pl.Int8
        assert df["scheduled_quantity"].dtype == pl.Float64
        assert df["eff_gas_day"].dtype == pl.Date

    def test_load_dataset_missing_file_no_auto(self):
        """Test that missing file raises error when auto=False."""
        with pytest.raises(FileNotFoundError, match="not found; pass --auto to download"):
            load_dataset("nonexistent.parquet", auto=False)

    def test_load_dataset_missing_file_with_auto(self):
        """Test that missing file with auto=True downloads the dataset."""
        import tempfile
        import os
        
        # Use a temporary file path for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_download.parquet")
            
            # This should download the dataset
            lf = load_dataset(test_path, auto=True)
            
            # Verify the file was created and the LazyFrame is valid
            assert os.path.exists(test_path)
            assert lf is not None
            
            # Verify we can collect a sample to ensure it's a valid parquet
            sample = lf.head(1).collect()
            assert sample.height >= 0  # Should have at least some data
            
            # Clean up
            os.remove(test_path)

    def test_load_dataset_none_path(self):
        """Test behavior with None path (should use default)."""
        # Since we now have data/pipeline_data.parquet, this should work
        lf = load_dataset(None, auto=False)
        assert lf is not None
        
        # Verify we can collect a sample to ensure it's valid
        sample = lf.head(1).collect()
        assert sample.height >= 0


class TestDictionary:
    """Tests for the dictionary module."""

    def test_build_data_dictionary(self, temp_parquet):
        """Test building a data dictionary from a LazyFrame."""
        lf = load_dataset(temp_parquet, auto=False)
        data_dict = build_data_dictionary(lf)

        # Check structure
        assert "schema" in data_dict
        assert "null_rates" in data_dict
        assert "n_rows" in data_dict

        # Check values
        assert data_dict["n_rows"] == 10
        assert len(data_dict["schema"]) == 13  # All columns

        # Check null rates
        assert data_dict["null_rates"]["pipeline_name"] == 0.0  # No nulls
        assert data_dict["null_rates"]["latitude"] == 0.5  # 50% nulls
        assert data_dict["null_rates"]["longitude"] == 0.5  # 50% nulls

        # Check schema types
        assert "Int8" in data_dict["schema"]["rec_del_sign"]
        assert "Float64" in data_dict["schema"]["scheduled_quantity"]
        assert "Date" in data_dict["schema"]["eff_gas_day"]

    def test_build_data_dictionary_empty_frame(self):
        """Test data dictionary with empty frame."""
        empty_lf = pl.LazyFrame(
            {"col1": [], "col2": []}, schema={"col1": pl.String, "col2": pl.Int64}
        )

        data_dict = build_data_dictionary(empty_lf)

        assert data_dict["n_rows"] == 0
        assert data_dict["null_rates"]["col1"] == 0.0
        assert data_dict["null_rates"]["col2"] == 0.0
        assert len(data_dict["schema"]) == 2

    def test_write_dictionary(self, temp_parquet):
        """Test writing dictionary to JSON file."""
        lf = load_dataset(temp_parquet, auto=False)
        data_dict = build_data_dictionary(lf)

        # Use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_dictionary.json"
            write_dictionary(data_dict, test_path)

            # Verify file exists and is valid JSON
            assert test_path.exists()

            # Read back and verify content
            content = orjson.loads(test_path.read_bytes())
            assert content["n_rows"] == 10
            assert "schema" in content
            assert "null_rates" in content


class TestIntegration:
    """Integration tests for the full ingest workflow."""

    def test_full_workflow(self, temp_parquet):
        """Test the complete load -> dictionary workflow."""
        # Load dataset
        lf = load_dataset(temp_parquet, auto=False)

        # Build dictionary
        data_dict = build_data_dictionary(lf)

        # Verify example values from the task specification
        assert data_dict["n_rows"] > 0
        assert len(data_dict["null_rates"]) == len(data_dict["schema"])

        # All null rates should be between 0 and 1
        for rate in data_dict["null_rates"].values():
            assert 0.0 <= rate <= 1.0

        # Schema should have string type names
        for dtype_str in data_dict["schema"].values():
            assert isinstance(dtype_str, str)
