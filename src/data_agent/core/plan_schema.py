"""Pydantic schemas for query plans."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Filter(BaseModel):
    """A filter operation on a column."""

    column: str
    op: Literal["=", "in", "between", "is_not_null", "contains"]
    value: Any


class Aggregate(BaseModel):
    """Aggregation operations with groupby and metrics."""

    groupby: list[str] = Field(default_factory=list)
    metrics: list[dict[str, str]] = Field(default_factory=list)  # {col, fn}


class Resample(BaseModel):
    """Time-based resampling configuration."""

    freq: str = "1d"
    on: str = "eff_gas_day"


class Sort(BaseModel):
    """Sorting configuration."""

    by: list[str]
    desc: bool = True
    limit: int | None = None


class Plan(BaseModel):
    """Complete query execution plan."""

    filters: list[Filter] = Field(default_factory=list)
    resample: Resample | None = None
    aggregate: Aggregate | None = None
    sort: Sort | None = None
    op: Literal["metric_compute", "changepoint", "cluster", "rules_scan", "causal", None] | None = (
        None
    )
    op_args: dict[str, Any] = Field(default_factory=dict)
    evidence: bool = True
    format: Literal["table", "json"] = "table"
