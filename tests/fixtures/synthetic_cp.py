"""
Synthetic data generation for changepoint detection testing.

This module generates time series data with known regime shifts/changepoints
for testing changepoint detection algorithms like PELT.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Union


def generate_changepoint_series(
    start_date: str,
    end_date: str,
    changepoints: List[str],
    regime_values: List[float],
    noise_std: float = 50.0,
    pipeline_names: Optional[List[str]] = None,
    trend_slope: float = 0.0,
    seed: int = 42
) -> pl.DataFrame:
    """
    Generate synthetic time series with known changepoints.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        changepoints: List of changepoint dates in YYYY-MM-DD format
        regime_values: List of mean values for each regime (length should be len(changepoints) + 1)
        noise_std: Standard deviation of random noise
        pipeline_names: List of pipeline names to generate data for
        trend_slope: Linear trend slope (units per day) applied across all regimes
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: date, pipeline_name, value, regime_id, true_changepoint
    """
    np.random.seed(seed)
    
    if pipeline_names is None:
        pipeline_names = ["TestPipeline1"]
    
    if len(regime_values) != len(changepoints) + 1:
        raise ValueError(f"regime_values length ({len(regime_values)}) must be len(changepoints) + 1 ({len(changepoints) + 1})")
    
    # Generate date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_range = []
    current_dt = start_dt
    while current_dt <= end_dt:
        date_range.append(current_dt)
        current_dt += timedelta(days=1)
    
    # Convert changepoint dates
    changepoint_dts = [datetime.strptime(cp, "%Y-%m-%d") for cp in changepoints]
    
    # Generate data for each pipeline
    all_data = []
    
    for i, pipeline_name in enumerate(pipeline_names):
        # Create slightly different patterns for each pipeline
        pipeline_seed = seed + i * 1000
        np.random.seed(pipeline_seed)
        
        # Generate values for each date
        for j, date in enumerate(date_range):
            # Determine which regime this date belongs to
            regime_id = 0
            for k, cp_date in enumerate(changepoint_dts):
                if date >= cp_date:
                    regime_id = k + 1
                else:
                    break
            
            # Base value for this regime
            base_value = regime_values[regime_id]
            
            # Add linear trend
            trend_value = trend_slope * j
            
            # Add noise
            noise = np.random.normal(0, noise_std)
            
            # Final value
            value = base_value + trend_value + noise
            
            # Ensure non-negative (gas quantities should be positive)
            value = max(value, 0)
            
            # Check if this date is a true changepoint
            is_changepoint = date.strftime("%Y-%m-%d") in changepoints
            
            all_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "pipeline_name": pipeline_name,
                "value": value,
                "regime_id": regime_id,
                "true_changepoint": is_changepoint
            })
    
    # Convert to Polars DataFrame
    df = pl.DataFrame(all_data)
    
    # Convert date column to proper date type
    df = df.with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    ])
    
    return df


def generate_multiple_changepoint_series(
    start_date: str,
    end_date: str,
    pipeline_configs: List[dict],
    seed: int = 42
) -> pl.DataFrame:
    """
    Generate multiple time series with different changepoint patterns.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        pipeline_configs: List of config dicts, each containing:
            - pipeline_name: str
            - changepoints: List[str]
            - regime_values: List[float]
            - noise_std: float (optional, default 50)
            - trend_slope: float (optional, default 0)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with all pipeline data combined
    """
    all_dfs = []
    
    for i, config in enumerate(pipeline_configs):
        pipeline_seed = seed + i * 1000
        
        df = generate_changepoint_series(
            start_date=start_date,
            end_date=end_date,
            changepoints=config["changepoints"],
            regime_values=config["regime_values"],
            noise_std=config.get("noise_std", 50.0),
            pipeline_names=[config["pipeline_name"]],
            trend_slope=config.get("trend_slope", 0.0),
            seed=pipeline_seed
        )
        
        all_dfs.append(df)
    
    # Combine all dataframes
    combined_df = pl.concat(all_dfs)
    
    return combined_df


def generate_gradual_changepoint_series(
    start_date: str,
    end_date: str,
    changepoint_date: str,
    initial_value: float,
    final_value: float,
    transition_days: int = 30,
    noise_std: float = 50.0,
    pipeline_names: Optional[List[str]] = None,
    seed: int = 42
) -> pl.DataFrame:
    """
    Generate time series with a gradual (not abrupt) changepoint.
    
    This tests the robustness of changepoint detection algorithms.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        changepoint_date: Date when transition begins
        initial_value: Mean value before transition
        final_value: Mean value after transition
        transition_days: Number of days over which transition occurs
        noise_std: Standard deviation of random noise
        pipeline_names: List of pipeline names to generate data for
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: date, pipeline_name, value, transition_progress
    """
    np.random.seed(seed)
    
    if pipeline_names is None:
        pipeline_names = ["TestPipeline1"]
    
    # Generate date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    changepoint_dt = datetime.strptime(changepoint_date, "%Y-%m-%d")
    
    date_range = []
    current_dt = start_dt
    while current_dt <= end_dt:
        date_range.append(current_dt)
        current_dt += timedelta(days=1)
    
    # Generate data for each pipeline
    all_data = []
    
    for i, pipeline_name in enumerate(pipeline_names):
        pipeline_seed = seed + i * 1000
        np.random.seed(pipeline_seed)
        
        for date in date_range:
            # Calculate transition progress (0 = before, 1 = after)
            days_from_start = (date - changepoint_dt).days
            
            if days_from_start < 0:
                # Before transition
                progress = 0.0
                base_value = initial_value
            elif days_from_start >= transition_days:
                # After transition
                progress = 1.0
                base_value = final_value
            else:
                # During transition - sigmoid curve
                progress = days_from_start / transition_days
                # Smooth sigmoid transition
                sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
                base_value = initial_value + (final_value - initial_value) * sigmoid_progress
            
            # Add noise
            noise = np.random.normal(0, noise_std)
            value = base_value + noise
            
            # Ensure non-negative
            value = max(value, 0)
            
            all_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "pipeline_name": pipeline_name,
                "value": value,
                "transition_progress": progress
            })
    
    # Convert to Polars DataFrame
    df = pl.DataFrame(all_data)
    
    # Convert date column to proper date type
    df = df.with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    ])
    
    return df


def validate_changepoint_detection(
    detected_changepoints: List[str],
    true_changepoints: List[str],
    tolerance_days: int = 7
) -> dict:
    """
    Validate changepoint detection results against known true changepoints.
    
    Args:
        detected_changepoints: List of detected changepoint dates (YYYY-MM-DD)
        true_changepoints: List of true changepoint dates (YYYY-MM-DD)
        tolerance_days: Tolerance in days for considering a detection correct
        
    Returns:
        Dict with validation metrics: precision, recall, f1_score, correct_detections
    """
    # Convert to datetime objects
    detected_dts = [datetime.strptime(cp, "%Y-%m-%d") for cp in detected_changepoints]
    true_dts = [datetime.strptime(cp, "%Y-%m-%d") for cp in true_changepoints]
    
    # Find correct detections (within tolerance)
    correct_detections = 0
    for true_cp in true_dts:
        for detected_cp in detected_dts:
            if abs((detected_cp - true_cp).days) <= tolerance_days:
                correct_detections += 1
                break
    
    # Calculate metrics
    precision = correct_detections / len(detected_changepoints) if detected_changepoints else 0
    recall = correct_detections / len(true_changepoints) if true_changepoints else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "correct_detections": correct_detections,
        "total_detected": len(detected_changepoints),
        "total_true": len(true_changepoints)
    }


if __name__ == "__main__":
    # Example usage - abrupt changepoints
    df = generate_changepoint_series(
        start_date="2020-01-01",
        end_date="2022-12-31",
        changepoints=["2020-06-01", "2021-03-01", "2021-09-01"],
        regime_values=[100, 200, 150, 300],
        noise_std=20,
        pipeline_names=["TestPipe1", "TestPipe2"]
    )
    
    print(f"Generated {len(df)} rows of synthetic changepoint data")
    print("Regime summary:")
    print(df.group_by("regime_id").agg([
        pl.col("value").mean().alias("mean_value"),
        pl.col("value").count().alias("count"),
        pl.col("date").min().alias("start_date"),
        pl.col("date").max().alias("end_date")
    ]).sort("regime_id"))
    
    # Example usage - gradual changepoint
    df_gradual = generate_gradual_changepoint_series(
        start_date="2020-01-01",
        end_date="2022-12-31",
        changepoint_date="2021-01-01",
        initial_value=100,
        final_value=300,
        transition_days=60,
        pipeline_names=["TestPipe1"]
    )
    
    print(f"\nGenerated {len(df_gradual)} rows with gradual changepoint")
    print("Transition progress summary:")
    print(df_gradual.group_by(
        pl.col("transition_progress").round(1)
    ).agg([
        pl.col("value").mean().alias("mean_value"),
        pl.col("value").count().alias("count")
    ]).sort("transition_progress"))
