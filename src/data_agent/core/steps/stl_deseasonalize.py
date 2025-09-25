"""STL deseasonalize step implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..handles import HandleStorage, StepHandle, StepStats, create_lazy_handle


def run(
    handle: StepHandle,
    params: dict[str, Any],
    storage: HandleStorage | None = None,
    dataset_digest: str = "",
) -> StepHandle:
    """Apply STL seasonal-trend decomposition with weekly + annual components.

    Args:
        handle: Input step handle
        params: STL parameters with format:
            {
                "column": str,              # column to deseasonalize
                "date_column": str,         # date column (default: "eff_gas_day")
                "seasonal": ["weekly", "annual"],  # seasonal components
                "trend_window": int,        # trend smoother window (optional)
                "seasonal_windows": dict,   # seasonal smoother windows (optional)
                "on": str                   # alias for column (for compatibility)
            }
        storage: Handle storage for materialization (optional)
        dataset_digest: Dataset digest for fingerprinting

    Returns:
        New step handle with deseasonalized data
    """
    # Load data from handle
    if handle.store == "parquet" and handle.path is not None:
        lf = pl.scan_parquet(handle.path)
    else:
        # For lazy handles, we assume the caller maintains the LazyFrame
        raise ValueError("STL deseasonalize step requires materialized input handle")

    # Extract parameters
    column = params.get("column") or params.get("on", "scheduled_quantity")
    date_column = params.get("date_column", "eff_gas_day")
    seasonal_components = params.get("seasonal", ["weekly", "annual"])
    trend_window = params.get("trend_window", 365)  # Default to annual trend window
    seasonal_windows = params.get("seasonal_windows", {})

    # Collect data for STL processing (STL requires full data in memory)
    df = lf.collect()

    if df.is_empty():
        return create_lazy_handle(
            f"{handle.id}_deseasonalized",
            schema=handle.schema,
            engine=handle.engine,
        )

    # Sort by date to ensure proper time series order
    df = df.sort(date_column)

    # Apply STL decomposition
    try:
        df_deseasonalized = _apply_stl_decomposition(
            df, column, date_column, seasonal_components, trend_window, seasonal_windows
        )
    except Exception as e:
        # Fallback: return original data with a warning column
        df_deseasonalized = df.with_columns(
            [
                pl.col(column).alias("deseasonalized"),
                pl.lit(f"STL failed: {str(e)}").alias("stl_warning"),
            ]
        )

    # Determine if we should materialize
    should_materialize = True  # STL is expensive, always materialize
    if storage is not None and should_materialize:
        # Compute fingerprint for caching
        input_fingerprints = [handle.fingerprint] if handle.fingerprint else []
        fingerprint = storage.compute_fingerprint(
            f"{handle.id}_deseasonalized",
            params,
            input_fingerprints,
            dataset_digest,
        )

        # Check if already cached
        cached_path = storage.get_storage_path(fingerprint)
        if cached_path.exists():
            # Load from cache
            df_cached = pl.read_parquet(cached_path)
            stats = StepStats(
                rows=df_cached.height,
                bytes=cached_path.stat().st_size,
                columns=df_cached.width,
                null_count={col: df_cached[col].null_count() for col in df_cached.columns},
                computed_at=cached_path.stat().st_mtime,
            )
            schema = {col: str(dtype) for col, dtype in zip(df_cached.columns, df_cached.dtypes)}

            return StepHandle(
                id=f"{handle.id}_deseasonalized",
                store="parquet",
                path=cached_path,
                engine=handle.engine,
                schema=schema,
                stats=stats,
                fingerprint=fingerprint,
            )
        else:
            # Materialize and cache
            return storage.materialize_handle(
                df_deseasonalized, fingerprint, f"{handle.id}_deseasonalized", handle.engine
            )
    else:
        # Return lazy handle (though data is already collected)
        new_schema = dict(handle.schema) if handle.schema else {}
        new_schema["deseasonalized"] = new_schema.get(column, "Float64")
        new_schema["trend"] = "Float64"
        new_schema["seasonal_weekly"] = "Float64"
        new_schema["seasonal_annual"] = "Float64"
        new_schema["residual"] = "Float64"

        return create_lazy_handle(
            f"{handle.id}_deseasonalized",
            schema=new_schema,
            engine=handle.engine,
        )


def _apply_stl_decomposition(
    df: pl.DataFrame,
    value_col: str,
    date_col: str,
    seasonal_components: list[str],
    trend_window: int,
    seasonal_windows: dict[str, int],
) -> pl.DataFrame:
    """Apply STL decomposition with multiple seasonal components.

    This is a simplified implementation. In production, you would use
    a proper STL library like statsmodels or implement LOESS smoothing.
    """
    # Extract values and dates
    values = df[value_col].to_numpy()
    dates = df[date_col].to_list()

    if len(values) < 14:  # Need at least 2 weeks of data
        # Return original data with zero seasonal components
        return df.with_columns(
            [
                pl.col(value_col).alias("deseasonalized"),
                pl.col(value_col).alias("trend"),
                pl.lit(0.0).alias("seasonal_weekly"),
                pl.lit(0.0).alias("seasonal_annual"),
                pl.lit(0.0).alias("residual"),
            ]
        )

    # Simple STL-like decomposition
    # 1. Compute trend using moving average
    trend = _compute_trend(values, trend_window)

    # 2. Detrend the data
    detrended = values - trend

    # 3. Extract seasonal components
    seasonal_weekly = np.zeros_like(values)
    seasonal_annual = np.zeros_like(values)

    if "weekly" in seasonal_components:
        seasonal_weekly = _extract_seasonal_component(detrended, dates, period=7)
        detrended = detrended - seasonal_weekly

    if "annual" in seasonal_components:
        seasonal_annual = _extract_seasonal_component(detrended, dates, period=365)
        detrended = detrended - seasonal_annual

    # 4. Residual is what's left
    residual = detrended

    # 5. Deseasonalized = original - all seasonal components
    deseasonalized = values - seasonal_weekly - seasonal_annual

    # Add the computed components to the dataframe
    return df.with_columns(
        [
            pl.Series("deseasonalized", deseasonalized),
            pl.Series("trend", trend),
            pl.Series("seasonal_weekly", seasonal_weekly),
            pl.Series("seasonal_annual", seasonal_annual),
            pl.Series("residual", residual),
        ]
    )


def _compute_trend(values: np.ndarray, window: int) -> np.ndarray:
    """Compute trend using centered moving average."""
    trend = np.zeros_like(values)
    half_window = window // 2

    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        trend[i] = np.mean(values[start:end])

    return trend


def _extract_seasonal_component(values: np.ndarray, dates: list, period: int) -> np.ndarray:
    """Extract seasonal component with given period."""
    seasonal = np.zeros_like(values)

    # Group by seasonal period and compute averages
    seasonal_means = {}
    seasonal_counts = {}

    for i, date in enumerate(dates):
        if isinstance(date, str):
            # Parse date string to get day of year or day of week
            try:
                import datetime

                parsed_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                parsed_date = datetime.datetime.now()
        else:
            parsed_date = date

        if period == 7:  # Weekly seasonality
            seasonal_key = parsed_date.weekday()
        elif period == 365:  # Annual seasonality
            seasonal_key = parsed_date.timetuple().tm_yday
        else:
            seasonal_key = i % period

        if seasonal_key not in seasonal_means:
            seasonal_means[seasonal_key] = 0.0
            seasonal_counts[seasonal_key] = 0

        seasonal_means[seasonal_key] += values[i]
        seasonal_counts[seasonal_key] += 1

    # Compute averages
    for key in seasonal_means:
        if seasonal_counts[key] > 0:
            seasonal_means[key] /= seasonal_counts[key]

    # Assign seasonal values
    for i, date in enumerate(dates):
        if isinstance(date, str):
            try:
                import datetime

                parsed_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                parsed_date = datetime.datetime.now()
        else:
            parsed_date = date

        if period == 7:  # Weekly seasonality
            seasonal_key = parsed_date.weekday()
        elif period == 365:  # Annual seasonality
            seasonal_key = parsed_date.timetuple().tm_yday
        else:
            seasonal_key = i % period

        seasonal[i] = seasonal_means.get(seasonal_key, 0.0)

    # Center the seasonal component (remove mean)
    seasonal = seasonal - np.mean(seasonal)

    return seasonal
