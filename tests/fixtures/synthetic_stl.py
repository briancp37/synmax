"""
Synthetic data generation for STL (Seasonal and Trend decomposition using Loess) testing.

This module generates time series data with known seasonal components (weekly and annual)
and trend patterns for testing STL decomposition functionality.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional


def generate_seasonal_series(
    start_date: str,
    end_date: str,
    weekly_amplitude: float = 100.0,
    annual_amplitude: float = 500.0,
    trend_slope: float = 0.1,
    noise_std: float = 50.0,
    pipeline_names: Optional[List[str]] = None,
    base_value: float = 1000.0,
    seed: int = 42
) -> pl.DataFrame:
    """
    Generate synthetic time series with known seasonal patterns.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        weekly_amplitude: Amplitude of weekly seasonal component
        annual_amplitude: Amplitude of annual seasonal component
        trend_slope: Linear trend slope (units per day)
        noise_std: Standard deviation of random noise
        pipeline_names: List of pipeline names to generate data for
        base_value: Base value around which to generate the series
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: date, pipeline_name, value
    """
    np.random.seed(seed)
    
    if pipeline_names is None:
        pipeline_names = ["TestPipeline1", "TestPipeline2", "TestPipeline3"]
    
    # Generate date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_range = []
    current_dt = start_dt
    while current_dt <= end_dt:
        date_range.append(current_dt)
        current_dt += timedelta(days=1)
    
    n_days = len(date_range)
    
    # Generate data for each pipeline
    all_data = []
    
    for i, pipeline_name in enumerate(pipeline_names):
        # Create slightly different patterns for each pipeline
        pipeline_seed = seed + i * 1000
        np.random.seed(pipeline_seed)
        
        # Base trend component
        trend = base_value + trend_slope * np.arange(n_days)
        
        # Weekly seasonal component (day of week effect)
        # Peak on weekdays, lower on weekends
        weekly_pattern = []
        for date in date_range:
            day_of_week = date.weekday()  # 0=Monday, 6=Sunday
            if day_of_week < 5:  # Weekday
                weekly_val = weekly_amplitude * np.sin(2 * np.pi * day_of_week / 7)
            else:  # Weekend
                weekly_val = -weekly_amplitude * 0.5
            weekly_pattern.append(weekly_val)
        
        weekly_seasonal = np.array(weekly_pattern)
        
        # Annual seasonal component (higher in winter, lower in summer)
        day_of_year = np.array([date.timetuple().tm_yday for date in date_range])
        annual_seasonal = annual_amplitude * np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
        
        # Add some pipeline-specific phase shifts
        pipeline_phase = i * np.pi / 4
        annual_seasonal = annual_amplitude * np.cos(2 * np.pi * (day_of_year - 1) / 365.25 + pipeline_phase)
        
        # Random noise
        noise = np.random.normal(0, noise_std, n_days)
        
        # Combine all components
        values = trend + weekly_seasonal + annual_seasonal + noise
        
        # Ensure no negative values (gas quantities should be positive)
        values = np.maximum(values, 0)
        
        # Create records for this pipeline
        for j, date in enumerate(date_range):
            all_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "pipeline_name": pipeline_name,
                "value": values[j]
            })
    
    # Convert to Polars DataFrame
    df = pl.DataFrame(all_data)
    
    # Convert date column to proper date type
    df = df.with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    ])
    
    return df


def generate_seasonal_series_with_regime_shift(
    start_date: str,
    end_date: str,
    shift_date: str,
    shift_magnitude: float,
    weekly_amplitude: float = 100.0,
    annual_amplitude: float = 500.0,
    trend_slope: float = 0.1,
    noise_std: float = 50.0,
    pipeline_names: Optional[List[str]] = None,
    base_value: float = 1000.0,
    seed: int = 42
) -> pl.DataFrame:
    """
    Generate synthetic time series with seasonal patterns and a regime shift.
    
    This is useful for testing combined STL + changepoint detection.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        shift_date: Date of regime shift in YYYY-MM-DD format
        shift_magnitude: Magnitude of the level shift
        weekly_amplitude: Amplitude of weekly seasonal component
        annual_amplitude: Amplitude of annual seasonal component
        trend_slope: Linear trend slope (units per day)
        noise_std: Standard deviation of random noise
        pipeline_names: List of pipeline names to generate data for
        base_value: Base value around which to generate the series
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: date, pipeline_name, value, regime_shift_flag
    """
    # Generate base seasonal series
    df = generate_seasonal_series(
        start_date=start_date,
        end_date=end_date,
        weekly_amplitude=weekly_amplitude,
        annual_amplitude=annual_amplitude,
        trend_slope=trend_slope,
        noise_std=noise_std,
        pipeline_names=pipeline_names,
        base_value=base_value,
        seed=seed
    )
    
    # Parse shift date
    shift_dt = datetime.strptime(shift_date, "%Y-%m-%d").date()
    
    # Apply regime shift
    df = df.with_columns([
        # Flag for regime shift
        (pl.col("date") >= shift_dt).alias("regime_shift_flag"),
        
        # Apply shift to values after shift_date
        pl.when(pl.col("date") >= shift_dt)
        .then(pl.col("value") + shift_magnitude)
        .otherwise(pl.col("value"))
        .alias("value")
    ])
    
    return df


def validate_stl_components(
    df: pl.DataFrame,
    expected_weekly_amplitude: float,
    expected_annual_amplitude: float,
    tolerance: float = 0.2
) -> bool:
    """
    Validate that STL decomposition correctly identified seasonal components.
    
    Args:
        df: DataFrame with STL components (trend, seasonal_weekly, seasonal_annual, residual)
        expected_weekly_amplitude: Expected amplitude of weekly component
        expected_annual_amplitude: Expected amplitude of annual component
        tolerance: Tolerance for amplitude comparison (as fraction)
        
    Returns:
        True if components are within expected ranges
    """
    required_cols = ["trend", "seasonal_weekly", "seasonal_annual", "residual"]
    
    # Check that all required columns exist
    for col in required_cols:
        if col not in df.columns:
            return False
    
    # Check weekly component amplitude
    weekly_amplitude = df.select(
        (pl.col("seasonal_weekly").max() - pl.col("seasonal_weekly").min()) / 2
    ).item()
    
    weekly_ok = (
        abs(weekly_amplitude - expected_weekly_amplitude) / expected_weekly_amplitude 
        <= tolerance
    )
    
    # Check annual component amplitude  
    annual_amplitude = df.select(
        (pl.col("seasonal_annual").max() - pl.col("seasonal_annual").min()) / 2
    ).item()
    
    annual_ok = (
        abs(annual_amplitude - expected_annual_amplitude) / expected_annual_amplitude
        <= tolerance
    )
    
    return weekly_ok and annual_ok


if __name__ == "__main__":
    # Example usage
    df = generate_seasonal_series(
        start_date="2020-01-01",
        end_date="2022-12-31",
        weekly_amplitude=200,
        annual_amplitude=800,
        pipeline_names=["TestPipe1", "TestPipe2"]
    )
    
    print(f"Generated {len(df)} rows of synthetic seasonal data")
    print(df.head(10))
    
    # Generate with regime shift
    df_shift = generate_seasonal_series_with_regime_shift(
        start_date="2020-01-01", 
        end_date="2022-12-31",
        shift_date="2021-06-01",
        shift_magnitude=500,
        pipeline_names=["TestPipe1"]
    )
    
    print(f"\nGenerated {len(df_shift)} rows with regime shift")
    print("Regime shift summary:")
    print(df_shift.group_by("regime_shift_flag").agg([
        pl.col("value").mean().alias("mean_value"),
        pl.col("value").count().alias("count")
    ]))
