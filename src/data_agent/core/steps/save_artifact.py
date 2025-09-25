"""Save artifact step implementation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import polars as pl

from ...config import ARTIFACTS_DIR
from ..handles import HandleStorage, StepHandle, create_lazy_handle


def run(
    handle: StepHandle,
    params: dict[str, Any],
    storage: HandleStorage | None = None,
    dataset_digest: str = "",
) -> StepHandle:
    """Save data as artifacts (tables/plots).

    Args:
        handle: Input step handle
        params: Save artifact parameters with format:
            {
                "format": str,              # output format: "parquet", "csv", "json"
                "filename": str,            # output filename (optional)
                "path": str,                # output path (optional, defaults to artifacts/)
                "include_plots": bool,      # whether to generate plots (default: False)
            }
        storage: Handle storage for materialization (optional)
        dataset_digest: Dataset digest for fingerprinting

    Returns:
        New step handle pointing to saved artifact
    """
    # Load data from handle
    if handle.store == "parquet" and handle.path is not None:
        lf = pl.scan_parquet(handle.path)
    else:
        # For lazy handles, we assume the caller maintains the LazyFrame
        raise ValueError("Save artifact step requires materialized input handle")

    # Extract parameters
    output_format = params.get("format", "parquet")
    filename = params.get("filename")
    output_path = params.get("path")
    include_plots = params.get("include_plots", False)

    # Collect data for saving
    df = lf.collect()

    # Determine output path
    if output_path:
        base_path = Path(output_path)
    else:
        base_path = ARTIFACTS_DIR / "outputs"

    base_path.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if not filename:
        timestamp = int(time.time())
        filename = f"artifact_{timestamp}_{handle.id}"

    # Save in requested format
    saved_files = []

    if output_format == "parquet":
        output_file = base_path / f"{filename}.parquet"
        df.write_parquet(output_file)
        saved_files.append(str(output_file))

    elif output_format == "csv":
        output_file = base_path / f"{filename}.csv"
        df.write_csv(output_file)
        saved_files.append(str(output_file))

    elif output_format == "json":
        output_file = base_path / f"{filename}.json"
        # Convert to JSON-serializable format
        data = {
            "data": df.to_dicts(),
            "schema": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            "metadata": {
                "rows": df.height,
                "columns": df.width,
                "source_handle": handle.id,
            },
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        saved_files.append(str(output_file))

    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    # Generate plots if requested
    if include_plots and df.height > 0:
        plot_files = _generate_plots(df, base_path, filename)
        saved_files.extend(plot_files)

    # Create metadata file
    metadata_file = base_path / f"{filename}_metadata.json"
    metadata = {
        "source_handle_id": handle.id,
        "source_fingerprint": handle.fingerprint,
        "output_format": output_format,
        "saved_files": saved_files,
        "rows": df.height,
        "columns": df.width,
        "column_names": df.columns,
        "column_types": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "created_at": str(time.time()),
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    saved_files.append(str(metadata_file))

    # Return handle pointing to the main saved file
    return create_lazy_handle(
        f"{handle.id}_saved",
        schema=handle.schema,
        engine=handle.engine,
    )


def _generate_plots(df: pl.DataFrame, output_path: Path, filename: str) -> list[str]:
    """Generate basic plots for the data.

    This is a simple implementation. In production, you might use
    matplotlib, plotly, or other visualization libraries.
    """
    plot_files: list[str] = []

    try:
        # Only generate plots if we have numeric columns
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype.is_numeric()]

        if not numeric_cols:
            return plot_files

        # Generate a simple summary plot (placeholder)
        # In a real implementation, you would create actual visualizations
        summary_data = {}
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            series = df[col]
            summary_data[col] = {
                "mean": float(series.mean()) if series.mean() is not None else 0.0,
                "std": float(series.std()) if series.std() is not None else 0.0,
                "min": float(series.min()) if series.min() is not None else 0.0,
                "max": float(series.max()) if series.max() is not None else 0.0,
                "count": int(series.count()),
            }

        # Save plot data as JSON (placeholder for actual plots)
        plot_file = output_path / f"{filename}_plot_data.json"
        with open(plot_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        plot_files.append(str(plot_file))

    except Exception as e:
        # If plot generation fails, just log it and continue
        error_file = output_path / f"{filename}_plot_error.txt"
        with open(error_file, "w") as f:
            f.write(f"Plot generation failed: {str(e)}")
        plot_files.append(str(error_file))

    return plot_files
