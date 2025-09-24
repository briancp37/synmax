"""Clustering functionality for fingerprinting entities by behavior patterns.

This module implements clustering of locations or counterparties based on their
gas flow patterns, including seasonality, ramps, reversals, and dependency metrics.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


def build_features(
    lf: pl.LazyFrame,
    entity_type: Literal["loc", "counterparty"] = "loc",
    min_days: int = 30,
) -> pl.DataFrame:
    """Build clustering features for entities.

    Args:
        lf: Input lazy frame with pipeline data
        entity_type: Type of entity to cluster - "loc" for locations,
            "counterparty" for counterparties
        min_days: Minimum number of days of data required for an entity

    Returns:
        DataFrame with features for each entity

    Features include:
    - Seasonality: 12-dimensional monthly normalized profiles
    - Ramps: Statistics on day-to-day changes (p95, std, mean)
    - Reversals: Frequency of flow direction changes
    - Dependency index: Herfindahl index of counterparty concentration
    - DOW bias: Day-of-week flow patterns
    """
    # Determine entity column based on type
    if entity_type == "loc":
        entity_col = "loc_name"
        counterparty_col = "connecting_entity"
    else:  # counterparty
        entity_col = "connecting_entity"
        counterparty_col = "loc_name"

    # First, create daily aggregates per entity
    daily_agg = (
        lf.with_columns(
            [
                pl.col("eff_gas_day").dt.month().alias("month"),
                pl.col("eff_gas_day").dt.weekday().alias("dow"),  # 1=Monday, 7=Sunday
            ]
        )
        .group_by([entity_col, "eff_gas_day", "month", "dow"])
        .agg(
            [
                pl.col("scheduled_quantity").sum().alias("daily_flow"),
                pl.col("rec_del_sign")
                .first()
                .alias("flow_direction"),  # Assuming consistent per day
                pl.col(counterparty_col).n_unique().alias("daily_counterparties"),
                pl.col(counterparty_col).first().alias("primary_counterparty"),
            ]
        )
    )

    # Filter entities with sufficient data
    entity_counts = (
        daily_agg.group_by(entity_col)
        .agg(pl.len().alias("n_days"))
        .filter(pl.col("n_days") >= min_days)
    )

    daily_filtered = daily_agg.join(entity_counts.select(entity_col), on=entity_col, how="inner")

    # Build features
    features = (
        daily_filtered.sort([entity_col, "eff_gas_day"])
        .with_columns(
            [
                # Calculate day-to-day changes for ramp analysis
                pl.col("daily_flow").diff().over(entity_col).alias("flow_diff"),
                # Lag for reversal detection
                pl.col("flow_direction").shift(1).over(entity_col).alias("prev_direction"),
            ]
        )
        .group_by(entity_col)
        .agg(
            [
                # Basic stats
                pl.len().alias("n_days"),
                pl.col("daily_flow").mean().alias("mean_flow"),
                pl.col("daily_flow").std().alias("std_flow"),
                pl.col("daily_flow").median().alias("median_flow"),
                # Seasonality features (monthly normalized)
                *[
                    (
                        pl.col("daily_flow").filter(pl.col("month") == m).mean()
                        / pl.col("daily_flow").mean().clip(1.0)
                    ).alias(f"season_m{m:02d}")
                    for m in range(1, 13)
                ],
                # Ramp features
                pl.col("flow_diff").abs().quantile(0.95).alias("ramp_p95"),
                pl.col("flow_diff").std().alias("ramp_std"),
                pl.col("flow_diff").mean().alias("ramp_mean"),
                # Reversal frequency
                (pl.col("flow_direction") != pl.col("prev_direction")).sum().alias("reversals"),
                # Day-of-week bias (coefficient of variation)
                *[
                    (
                        pl.col("daily_flow").filter(pl.col("dow") == d).mean()
                        / pl.col("daily_flow").mean().clip(1.0)
                    ).alias(f"dow_{d}")
                    for d in range(1, 8)
                ],
                # Dependency index (Herfindahl on counterparties)
                pl.col("primary_counterparty").n_unique().alias("n_counterparties"),
            ]
        )
        .with_columns(
            [
                # Normalize reversal frequency
                (pl.col("reversals") / pl.col("n_days").clip(1)).alias("reversal_freq"),
                # Fill missing seasonal values with 1.0 (average)
                *[pl.col(f"season_m{m:02d}").fill_null(1.0) for m in range(1, 13)],
                *[pl.col(f"dow_{d}").fill_null(1.0) for d in range(1, 8)],
                # Dependency index (inverse of counterparty count, normalized)
                (1.0 / pl.col("n_counterparties").clip(1)).alias("dependency_index"),
            ]
        )
        .collect()
    )

    return features


def cluster_entities(
    lf: pl.LazyFrame,
    entity_type: Literal["loc", "counterparty"] = "loc",
    k: int = 6,
    random_state: int = 42,
    min_days: int = 30,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Cluster entities by behavior patterns using KMeans.

    Args:
        lf: Input lazy frame with pipeline data
        entity_type: Type of entity to cluster
        k: Number of clusters
        random_state: Random seed for reproducibility
        min_days: Minimum days of data required per entity

    Returns:
        Tuple of (clustered_entities_df, metrics_dict)

    The DataFrame contains entity names, cluster labels, and cluster names.
    The metrics dict contains silhouette_score and other quality metrics.
    """
    # Build features
    features_df = build_features(lf, entity_type, min_days)

    if features_df.height == 0:
        raise ValueError(f"No entities found with at least {min_days} days of data")

    entity_col = "loc_name" if entity_type == "loc" else "connecting_entity"

    # Select feature columns for clustering
    feature_cols = [
        # Seasonality features
        *[f"season_m{m:02d}" for m in range(1, 13)],
        # Ramp features
        "ramp_p95",
        "ramp_std",
        "ramp_mean",
        # Reversal frequency
        "reversal_freq",
        # Dependency index
        "dependency_index",
        # Day-of-week features (simplified to weekday vs weekend)
        "dow_1",
        "dow_2",
        "dow_3",
        "dow_4",
        "dow_5",
        "dow_6",
        "dow_7",
    ]

    # Extract feature matrix
    feature_matrix = features_df.select(feature_cols).to_numpy()

    # Handle any remaining NaN values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Perform KMeans clustering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)

    # Calculate silhouette score
    # Silhouette score requires at least 2 clusters and fewer clusters than samples
    n_unique_labels = len(set(cluster_labels))
    if k > 1 and n_unique_labels > 1 and n_unique_labels < len(cluster_labels):
        silhouette = silhouette_score(feature_matrix_scaled, cluster_labels)
    else:
        silhouette = 0.0

    # Generate cluster names based on top z-scores
    cluster_names = _generate_cluster_names(feature_matrix_scaled, cluster_labels, feature_cols, k)

    # Create results DataFrame
    results_df = features_df.select([entity_col, "n_days", "mean_flow"]).with_columns(
        [
            pl.Series("cluster_id", cluster_labels),
            pl.Series("cluster_name", [cluster_names[label] for label in cluster_labels]),
        ]
    )

    # Compute metrics
    metrics = {
        "silhouette_score": float(silhouette),
        "n_entities": int(features_df.height),
        "n_clusters": int(k),
        "n_features": len(feature_cols),
        "min_cluster_size": int(np.bincount(cluster_labels).min()),
        "max_cluster_size": int(np.bincount(cluster_labels).max()),
    }

    return results_df, metrics


def _generate_cluster_names(
    feature_matrix: NDArray[np.float64],
    cluster_labels: NDArray[np.int32],
    feature_names: list[str],
    k: int,
) -> dict[int, str]:
    """Generate descriptive names for clusters based on top z-scores."""
    cluster_names = {}

    for cluster_id in range(k):
        mask = cluster_labels == cluster_id
        if not np.any(mask):
            cluster_names[cluster_id] = f"Empty_{cluster_id}"
            continue

        cluster_features = feature_matrix[mask]
        cluster_mean = np.mean(cluster_features, axis=0)

        # Calculate z-scores relative to overall population
        overall_mean = np.mean(feature_matrix, axis=0)
        overall_std = np.std(feature_matrix, axis=0)

        # Avoid division by zero
        z_scores = np.divide(
            cluster_mean - overall_mean,
            overall_std,
            out=np.zeros_like(overall_mean),
            where=overall_std != 0,
        )

        # Find top distinctive features
        top_indices = np.argsort(np.abs(z_scores))[-3:][::-1]  # Top 3 by absolute z-score

        name_parts = []
        for idx in top_indices:
            if abs(z_scores[idx]) > 0.5:  # Only include meaningful differences
                feature_name = feature_names[idx]
                direction = "High" if z_scores[idx] > 0 else "Low"

                # Simplify feature names
                if feature_name.startswith("season_m"):
                    month = int(feature_name.split("_m")[1])
                    month_name = [
                        "",
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ][month]
                    name_parts.append(f"{direction}_{month_name}")
                elif feature_name.startswith("dow_"):
                    day = int(feature_name.split("_")[1])
                    day_name = ["", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day]
                    name_parts.append(f"{direction}_{day_name}")
                elif "ramp" in feature_name:
                    name_parts.append(f"{direction}_Ramp")
                elif "reversal" in feature_name:
                    name_parts.append(f"{direction}_Reversals")
                elif "dependency" in feature_name:
                    name_parts.append(f"{direction}_Dependency")

        if name_parts:
            cluster_names[cluster_id] = "_".join(name_parts[:2])  # Limit to 2 parts
        else:
            cluster_names[cluster_id] = f"Cluster_{cluster_id}"

    return cluster_names


def run_cluster_analysis(
    lf: pl.LazyFrame,
    entity_type: Literal["loc", "counterparty"] = "loc",
    k: int = 6,
    random_state: int = 42,
) -> pl.DataFrame:
    """Run clustering analysis and return results table.

    This is the main entry point for the clustering operation,
    designed to be called from the executor when op="cluster".

    Args:
        lf: Input lazy frame
        entity_type: Type of entity to cluster
        k: Number of clusters
        random_state: Random seed

    Returns:
        DataFrame with clustering results suitable for display
    """
    results_df, metrics = cluster_entities(lf, entity_type, k, random_state)

    # Create summary table
    summary = (
        results_df.group_by(["cluster_id", "cluster_name"])
        .agg(
            [
                pl.len().alias("count"),
                pl.col("mean_flow").mean().alias("avg_flow"),
                pl.col("mean_flow").std().alias("std_flow"),
                pl.col("n_days").mean().alias("avg_days"),
            ]
        )
        .sort("cluster_id")
        .with_columns(
            [
                pl.lit(f"Silhouette: {metrics['silhouette_score']:.3f}").alias("quality"),
            ]
        )
    )

    return summary
