# Build Plan for Cursor


## Conventions

* Use **Polars** for core execution; optionally DuckDB for heavy groupbys.
* Typing enforced (`mypy --strict` where feasible).
* Run `make format test` before each PR.
* Always use poetry instead of pip.

---

## Task 0: Bootstrap repo

**Files:** `pyproject.toml`, `.gitignore`, `.env.example`, `Makefile`, `src/data_agent/__init__.py`

**Steps**

* Add dependencies: `polars`, `pyarrow`, `duckdb`, `typer`, `pydantic`, `python-dotenv`, `orjson`, `ruamel.yaml`, `numpy`, `scikit-learn`, `ruptures`, `tdigest`, `rich`, `pytest`, `pytest-cov`, `mypy`, `ruff`, `black`.
* Configure `ruff`, `black`, `mypy` in `pyproject.toml`.
* Add `Makefile` targets: `format`, `lint`, `typecheck`, `test`, `run`.
* `.gitignore`: `/data/`, `/artifacts/`, `/.cache/`, `.env`.

**Acceptance**

* `make format lint typecheck test` runs without error (empty tests OK).
* `poetry install` works.

**Code stub** (pyproject snippet):

```toml
[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
select = ["E","F","I","UP","B","N","Q"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_unused_ignores = true
```

---

## Task 1: CLI scaffold

**Files:** `src/data_agent/cli.py`, `src/data_agent/utils/logging.py`, `src/data_agent/config.py`

**Steps**

* Add Typer CLI with commands: `load`, `ask`, `rules`, `metrics`, `events`, `cluster`, `cache`.
* Centralize config and paths; ensure `ARTIFACTS_DIR` and `CACHE_DIR` creation.
* Structured logging helper.

**Acceptance**

* `python -m data_agent.cli --help` shows commands.
* `agent --help` works via console script if configured.

**Code stub** (`cli.py`):

```python
import typer
from data_agent import config

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command()
def load(path: str = typer.Option(None), auto: bool = False):
    """Load dataset from --path or auto-download to ./data."""
    # TODO: call ingest.loader.load_dataset
    typer.echo("Loaded dataset (placeholder).")

@app.command()
def ask(q: str, planner: str = typer.Option("deterministic", help="deterministic|llm"), export: str|None = None):
    """Ask a natural-language question about the dataset."""
    # TODO: planner.plan(q) -> Plan; executor.run(Plan) -> Answer+Evidence
    typer.echo("Answer (placeholder)")

if __name__ == "__main__":
    app()
```

---

## Task 2: Ingest: loader & data dictionary

**Files:** `src/data_agent/ingest/loader.py`, `src/data_agent/ingest/dictionary.py`

**Steps**

* Implement `load_dataset(path: str|None, auto: bool) -> pl.LazyFrame`.
* Auto-download if `auto=True` and `path` None; save to `./data/data.parquet`.
* Normalize dtypes: `eff_gas_day` → Date, `rec_del_sign` → Int8, `scheduled_quantity` → Float64.
* Implement `build_data_dictionary(lf: pl.LazyFrame) -> dict` and write JSON to `artifacts/data_dictionary.json`.

**Acceptance**

* `agent load --path examples/golden.parquet` prints schema + writes `artifacts/data_dictionary.json`.
* Unit tests validate null rates & example values presence.

**Code stubs**

```python
# loader.py
from __future__ import annotations
import polars as pl
from pathlib import Path

DEFAULT_DATA_PATH = Path("data/data.parquet")

def load_dataset(path: str | None, auto: bool) -> pl.LazyFrame:
    """Return a Polars LazyFrame for the dataset. Does not collect."""
    p = Path(path) if path else DEFAULT_DATA_PATH
    if not p.exists():
        if not auto:
            raise FileNotFoundError(f"{p} not found; pass --auto to download")
        # TODO: download via requests/gdown into p
    lf = pl.scan_parquet(str(p))
    # dtype normalization
    casts = {
        "rec_del_sign": pl.Int8,
        "scheduled_quantity": pl.Float64,
    }
    # eff_gas_day -> Date; handle string or date
    if "eff_gas_day" in lf.columns:
        lf = lf.with_columns(pl.col("eff_gas_day").str.strptime(pl.Date, strict=False).alias("eff_gas_day"))
    lf = lf.with_columns([pl.col(k).cast(v, strict=False) for k, v in casts.items() if k in lf.columns])
    return lf
```

```python
# dictionary.py
from __future__ import annotations
import polars as pl
import orjson
from pathlib import Path

ARTIFACTS = Path("artifacts"); ARTIFACTS.mkdir(parents=True, exist_ok=True)

def build_data_dictionary(lf: pl.LazyFrame) -> dict:
    df = lf.limit(0).collect_schema()
    schema = {name: str(dtype) for name, dtype in zip(df.names(), df.dtypes())}
    # compute null rates and n rows
    head = lf.select([pl.len().alias("n_rows")]).collect()
    n_rows = int(head[0, "n_rows"]) if head.height else 0
    nulls = (
        lf.select([pl.col(c).is_null().sum().alias(c) for c in lf.columns])
        .with_columns(pl.all().cast(pl.Float64) / max(n_rows, 1))
        .collect()
    )
    null_rates = {c: float(nulls[0, c]) for c in lf.columns}
    return {"schema": schema, "null_rates": null_rates, "n_rows": n_rows}

def write_dictionary(dic: dict, path: Path = ARTIFACTS / "data_dictionary.json") -> None:
    path.write_bytes(orjson.dumps(dic, option=orjson.OPT_INDENT_2))
```

---

## Task 3: Daily rollups (materialized view)

**Files:** `src/data_agent/ingest/loader.py` (extend), `src/data_agent/ingest/rollups.py`

**Steps**

* Create `build_daily_rollups(lf) -> pl.DataFrame` grouped by `(eff_gas_day, pipeline_name, state_abb, category_short)`.
* Metrics: `sum_receipts`, `sum_deliveries`, `sum_all`, `net`.
* Write to `artifacts/rollups/daily.parquet`.

**Acceptance**

* Golden test: deterministic output for `examples/golden.parquet` matches `examples/expected/daily.json` snapshot (or hash).
* CLI `agent load` prints path to rollups file.

**Code stub**

```python
# rollups.py
from __future__ import annotations
import polars as pl
from pathlib import Path

ROLLUP_DIR = Path("artifacts/rollups"); ROLLUP_DIR.mkdir(parents=True, exist_ok=True)

GROUP = ["eff_gas_day","pipeline_name","state_abb","category_short"]

def build_daily_rollups(lf: pl.LazyFrame) -> pl.DataFrame:
    receipts = (pl.col("rec_del_sign") == 1).cast(pl.Int8)
    deliveries = (pl.col("rec_del_sign") == -1).cast(pl.Int8)
    agg = (
        lf.group_by(GROUP)
        .agg([
            (pl.when(receipts==1).then(pl.col("scheduled_quantity")).otherwise(0.0)).sum().alias("sum_receipts"),
            (pl.when(deliveries==1).then(pl.col("scheduled_quantity")).otherwise(0.0)).sum().alias("sum_deliveries"),
            pl.col("scheduled_quantity").sum().alias("sum_all"),
        ])
        .with_columns((pl.col("sum_receipts")-pl.col("sum_deliveries")).alias("net"))
    )
    return agg.collect()

def write_daily_rollups(df: pl.DataFrame) -> Path:
    out = ROLLUP_DIR / "daily.parquet"
    df.write_parquet(out)
    return out
```

---

## Task 4: Plan schema + deterministic templates

**Files:** `src/data_agent/core/plan_schema.py`, `src/data_agent/core/planner.py`

**Steps**

* Define Pydantic classes: `Filter`, `Aggregate`, `Resample`, `Sort`, `Plan`.
* Deterministic template mapper: regex/rule-based map from a few NL forms → `Plan`.
* Add `planner.plan(q: str, deterministic=True) -> Plan`.

**Acceptance**

* Unit tests: 5 prompts produce valid `Plan` instances and expected JSON.
* CLI: `agent ask "sum of deliveries for ANR on 2022-01-01" --planner deterministic --dry-run` prints Plan JSON.

**Code stub**

```python
# plan_schema.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Literal

class Filter(BaseModel):
    column: str
    op: Literal["=","in","between","is_not_null","contains"]
    value: Any

class Aggregate(BaseModel):
    groupby: list[str] = Field(default_factory=list)
    metrics: list[dict] = Field(default_factory=list)  # {col, fn}

class Resample(BaseModel):
    freq: str = "1d"
    on: str = "eff_gas_day"

class Sort(BaseModel):
    by: list[str]
    desc: bool = True
    limit: int | None = None

class Plan(BaseModel):
    filters: list[Filter] = Field(default_factory=list)
    resample: Resample | None = None
    aggregate: Aggregate | None = None
    sort: Sort | None = None
    op: Literal["metric_compute","changepoint","cluster","rules_scan",None] | None = None
    op_args: dict = Field(default_factory=dict)
    evidence: bool = True
    format: Literal["table","json"] = "table"
```

```python
# planner.py
import re
from .plan_schema import Plan, Filter, Aggregate

_PATTERNS = [
    # sum deliveries for pipeline on date
    (re.compile(r"sum of deliveries for (.+) on (\d{4}-\d{2}-\d{2})", re.I), "sum_deliveries_on_date"),
    (re.compile(r"top (\d+) states by total scheduled quantity in (\d{4})", re.I), "top_states_year"),
]

def plan(q: str, deterministic: bool = True) -> Plan:
    for rx, key in _PATTERNS:
        m = rx.search(q)
        if not m:
            continue
        if key == "sum_deliveries_on_date":
            pipeline, d = m.group(1).strip(), m.group(2)
            return Plan(
                filters=[
                    Filter(column="pipeline_name", op="=", value=pipeline),
                    Filter(column="rec_del_sign", op="=", value=-1),
                    Filter(column="eff_gas_day", op="between", value=[d, d]),
                ],
                aggregate=Aggregate(groupby=[], metrics=[{"col":"scheduled_quantity","fn":"sum"}]),
            )
        if key == "top_states_year":
            n, year = int(m.group(1)), m.group(2)
            return Plan(
                filters=[Filter(column="eff_gas_day", op="between", value=[f"{year}-01-01", f"{year}-12-31"])],
                aggregate=Aggregate(groupby=["state_abb"], metrics=[{"col":"scheduled_quantity","fn":"sum"}]),
                sort={"by":["sum_scheduled_quantity"],"desc":True,"limit":n},
            )
    # Fallback minimal plan (could raise)
    return Plan()
```

---

## Task 5: Executor + Evidence Card (vertical slice)

**Files:** `src/data_agent/core/ops.py`, `src/data_agent/core/executor.py`, `src/data_agent/core/evidence.py`

**Steps**

* Implement ops: filter, resample, aggregate, sort/limit.
* Executor: accept `Plan`, run against `lf`, produce `Answer{table}` + `Evidence{json}`.
* Evidence: dataset digest, filters, rows in/out, missingness, runtime, repro snippet.

**Acceptance**

* Integration test on golden for two queries; match exact numbers.
* CLI `agent ask ...` prints table + evidence.

**Code stubs**

```python
# ops.py
from __future__ import annotations
import polars as pl
from .plan_schema import Plan

def apply_plan(lf: pl.LazyFrame, plan: Plan) -> pl.LazyFrame:
    out = lf
    for f in plan.filters:
        if f.op == "=":
            out = out.filter(pl.col(f.column) == f.value)
        elif f.op == "between":
            lo, hi = f.value
            out = out.filter((pl.col(f.column) >= lo) & (pl.col(f.column) <= hi))
        elif f.op == "in":
            out = out.filter(pl.col(f.column).is_in(f.value))
        elif f.op == "is_not_null":
            out = out.filter(pl.col(f.column).is_not_null())
        elif f.op == "contains":
            out = out.filter(pl.col(f.column).cast(pl.Utf8).str.contains(str(f.value)))
    if plan.aggregate:
        gb = plan.aggregate.groupby
        aggs = []
        for m in plan.aggregate.metrics:
            col, fn = m["col"], m["fn"].lower()
            if fn == "sum":
                aggs.append(pl.col(col).sum().alias(f"sum_{col}"))
            elif fn == "count":
                aggs.append(pl.len().alias("count"))
            elif fn == "avg":
                aggs.append(pl.col(col).mean().alias(f"avg_{col}"))
            elif fn == "p95":
                aggs.append(pl.col(col).quantile(0.95).alias(f"p95_{col}"))
            elif fn == "p50":
                aggs.append(pl.col(col).median().alias(f"p50_{col}"))
        out = out.group_by(gb).agg(aggs) if gb else out.select(aggs)
    # Sort/limit handled by executor after collect
    return out
```

```python
# executor.py
from __future__ import annotations
import time
import polars as pl
from .plan_schema import Plan
from . import ops
from .evidence import build_evidence

class Answer:
    def __init__(self, table: pl.DataFrame, evidence: dict):
        self.table = table
        self.evidence = evidence

def run(lf: pl.LazyFrame, plan: Plan) -> Answer:
    t0 = time.perf_counter()
    out_lf = ops.apply_plan(lf, plan)
    t1 = time.perf_counter()
    df = out_lf.collect()
    # sort/limit
    if plan.sort:
        by = plan.sort["by"] if isinstance(plan.sort, dict) else plan.sort.by
        desc = plan.sort.get("desc", True) if isinstance(plan.sort, dict) else plan.sort.desc
        limit = plan.sort.get("limit") if isinstance(plan.sort, dict) else plan.sort.limit
        df = df.sort(by=by, descending=desc)
        if limit:
            df = df.head(limit)
    t2 = time.perf_counter()
    ev = build_evidence(lf, plan, df, timings={"plan": t1 - t0, "collect": t2 - t1})
    return Answer(df, ev)
```

```python
# evidence.py
from __future__ import annotations
import hashlib
import polars as pl
from pathlib import Path
from .plan_schema import Plan

def _digest(parquet_path: Path) -> str:
    # light-weight digest: size+mtime
    try:
        st = parquet_path.stat()
        raw = f"{parquet_path}|{st.st_size}|{int(st.st_mtime)}".encode()
        return hashlib.sha256(raw).hexdigest()[:12]
    except FileNotFoundError:
        return "unknown"

def build_evidence(lf: pl.LazyFrame, plan: Plan, df: pl.DataFrame, timings: dict) -> dict:
    # Note: LazyFrame doesn't expose path; pass via config or set on context
    cols_used = sorted({f.column for f in plan.filters} | {m["col"] for m in (plan.aggregate.metrics if plan.aggregate else [])})
    miss = {c: None for c in cols_used}  # TODO compute null rates cheaply
    ev = {
        "filters": [f.model_dump() for f in plan.filters],
        "aggregate": plan.aggregate.model_dump() if plan.aggregate else None,
        "rows_out": int(df.height),
        "columns": list(df.columns),
        "missingness": miss,
        "timings_ms": {k: round(v*1000,1) for k,v in timings.items()},
        "cache": {"hit": False},
        "repro": {"engine":"polars","snippet":"# TODO generated"},
    }
    return ev
```

---

## Task 6: Rules engine

**Files:** `src/data_agent/rules/engine.py`

**Steps**

* Implement R‑001..R‑006 predicates; return dict of `{rule_id: {count:int, samples:list[dict]}}`.
* Wire into Evidence builder (include summary counts and up to 3 samples per rule).

**Acceptance**

* Unit tests: synthetic frames trigger each rule.
* CLI: `agent rules --pipeline ANR --since 2022-01-01` prints table with counts.

---

## Task 7: LLM planner (function-calling)

**Files:** `src/data_agent/core/planner.py` (extend)

**Steps**

* Add `--planner llm` path with OpenAI/Anthropic function schema equal to `Plan`. `.env` contains the ANTHROPIC_API_KEY and OPENAI_API_KEY.
* Make it interoperable between either OpenAI or Anthropic.
* Validate received JSON strictly; fallback to deterministic if invalid.

**Acceptance**

* Mocked tests feed a fixed JSON → valid Plan → executor runs.

---

## Task 8: Caching & performance

**Files:** `src/data_agent/cache/cache.py`, `src/data_agent/core/executor.py` (wire)

**Steps**

* Compute fingerprint from canonical Plan JSON + dataset digest.
* Store result DataFrame + evidence JSON to `./.cache/results/{hash}.parquet/json`.
* Evidence shows cache status & TTL.

**Acceptance**

* Repeated query returns `cache.hit=True` and faster elapsed time in test.

---

## Task 9: Analytics: metrics pack

**Files:** `src/data_agent/core/ops.py` (extend), `src/data_agent/core/metrics.py`

**Steps**

* Implement `ramp_risk`, `reversal_freq`, `imbalance_pct` functions that can be called via `op="metric_compute"` with `op_args.name`.
* Respect current filters; output tidy tables with well-named columns.

**Acceptance**

* Unit tests on toy series; integration on golden with snapshot tables.

---

## Task 10: Change-points + Event Cards

**Files:** `src/data_agent/core/events.py`, `src/data_agent/core/ops.py` (route)

**Steps**

* `changepoint` op: run ruptures PELT on selected aggregates; return event table.
* Build Event Card (before/after stats; contributors via diff of groupby means).

**Acceptance**

* Synthetic test with known breakpoints is correctly identified (±1 day tolerance).

---

## Task 10.1: 

The dataset is located on Google Drive at https://drive.google.com/file/d/109vhmnSLN3oofjFdyb58l_rUZRa0d6C8/view?usp=drivesdk
The system should check if the parquet is located in ./data/ and called data/pipeline_data.parquet.  If it is not, then the system should automatically download the data from the Google Drive link https://drive.google.com/file/d/109vhmnSLN3oofjFdyb58l_rUZRa0d6C8/view?usp=drivesdk and place it at data/pipeline_data.parquet.  

---

## Task 11: Clustering fingerprints

**Files:** `src/data_agent/core/cluster.py`

**Steps**

* Feature builder (seasonality, ramps, reversals, dependency index).
* KMeans with seed; labeler that names clusters via top z-scores.

**Acceptance**

* Reproducible clusters on golden; silhouette score > 0.2 (toy threshold).

---

## Task 12: Polish & README & export

**Files:** `README.md`, `examples/queries.md`, `src/data_agent/core/export.py`

**Steps**

* README: install, dataset path/auto-download, example commands, env vars, caveats.
* `--export` writes JSON answer+evidence to `artifacts/outputs/{run_id}.json`.

**Acceptance**

* Fresh clone quickstart completes; example queries produce expected shapes.

---







## Task 13: Causal Hypothesis Generation Module (core missing)

**Files:** `src/data_agent/core/causal.py`, `src/data_agent/core/planner.py` (extend), `src/data_agent/core/plan_schema.py` (extend), `tests/test_causal.py`

**Goal:** Given a question + filtered dataset (and optionally a detected pattern/event), produce **plausible causal hypotheses** with **supporting evidence** and **caveats**. LLM generates text; the module computes structured *evidence features* the LLM cites.

**Plan Schema Extension**

```python
# plan_schema.py additions
from typing import Literal

class CausalArgs(BaseModel):
    target: Literal["volume","imbalance","ramp_risk","reversal_freq"] = "volume"
    explain_scope: Literal["pipeline","state","corridor","entity"] = "pipeline"
    compare_to: Literal["peers","prior_period","seasonal_baseline"] = "prior_period"
    top_k_factors: int = 5

# Plan.op now supports "causal"
# Plan.op_args may be CausalArgs
```

**Module Design (**\`\`**)**

* Compute **diagnostic features** inside current filters:

  * **Seasonal deviation**: detrend with rolling 30d mean; z-score of the current period.
  * **Peer contrast**: percentile rank vs peers (same `state_abb` or peer pipelines).
  * **Balance residual**: imbalance% contemporaneous.
  * **Counterparty shift**: HHI change and top contributor deltas.
  * **Calendar signals**: weekend/holiday proximity (no external data — weekend boolean only).
* Create a **Causal Evidence JSON** that LLM will use to draft hypotheses.
* Prompt LLM with instruction to produce: 3–5 hypotheses, each with **mechanism**, **evidence bullets (from JSON)**, **confidence (low/med/high)**, and **caveats/alternative explanations**.

**LLM Prompt (template)**

```text
System: You are a cautious energy analyst. Propose plausible causes; never assert.
User JSON (Evidence): {CAUSAL_EVIDENCE_JSON}
Task: Produce 3–5 hypotheses explaining the observed relationship/pattern. For each:
- Mechanism (1 sentence)
- Evidence (2–4 bullets, cite numbers from evidence JSON)
- Confidence: low|medium|high (justify briefly)
- Caveats (1–2 bullets)
Return as JSON list of objects: {"mechanism","evidence":[...],"confidence","caveats":[...]}
```

**Executor Integration**

* If `Plan.op == "causal"`: compute `CausalEvidence`, call LLM (if key present), parse JSON, and embed into **Enhanced Evidence Card**.
* Deterministic fallback (no LLM): print structured evidence + a note: "LLM not enabled; presenting diagnostic factors only."

**Acceptance**

* Unit: synthetic input yields Evidence JSON with keys present and stable shapes.
* Mocked LLM: returns fixed JSON → parsed into `answer.hypotheses` with 3–5 entries.
* CLI: `agent ask "why did ANR deliveries spike in Jan 2022?" --planner llm` prints hypotheses with confidences and caveats.

**Code stub**

```python
# causal.py
from __future__ import annotations
import polars as pl
from dataclasses import dataclass
from .plan_schema import Plan

@dataclass
class CausalEvidence:
    stats: dict
    peers: dict
    balance: dict
    counterparties: dict
    calendar: dict

def build_causal_evidence(lf: pl.LazyFrame, plan: Plan) -> CausalEvidence:
    # TODO: compute detrended z, peer ranks, imbalance%, HHI deltas, weekend flag
    return CausalEvidence(stats={}, peers={}, balance={}, counterparties={}, calendar={})

def draft_hypotheses(evidence: CausalEvidence, llm_client=None) -> list[dict]:
    # If llm_client is None, return [] and rely on deterministic evidence
    return []
```

---

Task 13.1: 
- Add a linux flag so I can specify the llm model for the causal hypothesis generation. 
- Also and i need the ability to set a default model within the cli that is used unless i specify another one for a query.



---


## Task 14: Correlation Analysis (patterns: correlations & trends)

**Files:** `src/data_agent/core/correlation.py`, `src/data_agent/core/planner.py` (extend), `src/data_agent/core/plan_schema.py` (extend), `tests/test_correlation.py`

**Goal:** Compute correlations within the dataset with statistical rigor and clear caveats.

**Features**

* **Cross-variable correlations**: e.g., between `scheduled_quantity` and net metrics across groups.
* **Time-series correlations**: Pearson/Spearman across aligned daily series (pairwise groups).
* **Correlation matrix** for selected measures; optional **heatmap PNG** to `artifacts/`.
* **Significance testing**: p-values (with multiple-test FDR correction via Benjamini–Hochberg).

**Plan Schema Extension**

```python
class CorrelationArgs(BaseModel):
    level: Literal["pipeline","state","entity","corridor"] = "state"
    measure: Literal["sum","net","receipts","deliveries"] = "sum"
    method: Literal["pearson","spearman"] = "pearson"
    min_days: int = 60
    top_k_pairs: int = 25

# Plan.op supports "correlation" with CorrelationArgs
```

**Acceptance**

* Unit: known toy series produce expected Pearson ≈ 1.0 / 0.0 correlations.
* Integration: golden subset generates matrix with shape > 1×1, includes p-values and q-values.
* CLI: `agent ask "state-level correlations of daily sums in 2022 (pearson)"` prints top pairs with r, p, q.

**Code stub**

```python
# correlation.py
from __future__ import annotations
import polars as pl
import numpy as np
from typing import Literal

def pairwise_ts_correlations(df: pl.DataFrame, group_col: str, value_col: str, method: Literal["pearson","spearman"]="pearson") -> pl.DataFrame:
    # df: columns [date, group, value]
    # Align by date, pivot wider, then compute corr + p-values
    wide = df.pivot(index="eff_gas_day", columns=group_col, values=value_col)
    np_mat = wide.select(pl.all().exclude("eff_gas_day")).to_numpy()
    # TODO: compute correlation matrix and p-values
    return pl.DataFrame()
```

---

## Task 15: Enhanced Evidence with Causal Context

**Files:** `src/data_agent/core/evidence.py` (extend), `src/data_agent/core/causal.py` (integrate), `tests/test_evidence_enhanced.py`

**Goal:** Extend the Evidence Card to include causal context and confidence.

**Additions to Evidence Card**

* `pattern_summary`: brief description of detected pattern (e.g., spike, trend, cluster label) if relevant.
* `diagnostics`: structured metrics used in causal drafting (z-scores, peer ranks, imbalance%).
* `hypotheses`: list of {`mechanism`,`evidence` bullets, `confidence`, `caveats`}.
* `confidence_overall`: heuristic rollup (min of data-quality confidence and hypothesis confidence median).

**Acceptance**

* Unit: evidence builder merges causal artifacts when present.
* Integration: running a causal query produces hypotheses embedded in evidence JSON.
* CLI printout shows a new **“Causal Context”** section.

**Code stub**

```python
# evidence.py (excerpt)
ev["causal"] = {
  "pattern_summary": pattern_desc,
  "diagnostics": evidence.stats,
  "hypotheses": hypotheses,
  "confidence_overall": confidence,
}
```

---

## Task 16: Advanced Analytics Integration into NL Flow

**Files:** `src/data_agent/core/planner.py` (extend templates + llm intent), `src/data_agent/core/executor.py` (route ops), `tests/test_nl_integration.py`

**Goal:** Ensure NL questions can request causal analysis and correlations seamlessly.

**Deterministic Templates**

* "**why** did   change in ?" → `op="causal"` with proper scope.
* "**correlate**  by  in  using \<pearson|spearman>" → `op="correlation"`.

**LLM Planner**

* Update function schema examples to include `causal` and `correlation` plans.
* Guardrails if both ops requested: prioritize `correlation` first, then optional `causal` on top pairs.

**Acceptance**

* Unit: 6 new NL patterns → valid `Plan`.
* E2E: CLI queries for causal and correlation return sensible tables + enhanced evidence.

---

## Task 17: Business Interpretation Layer

**Files:** `src/data_agent/core/business.py`, `src/data_agent/core/cluster.py` (hook), `tests/test_business.py`

**Goal:** Translate technical outputs (clusters, metrics, events) into **actionable, business-friendly narratives** and **risk tags**.

**Features**

* **Cluster labeling**: Map feature z-scores to labels like `Peaky Winter LDC`, `Stable Transit Hub`, `Volatile Interconnect`.
* **Risk assessment**: Derive badges from metrics (e.g., High ramp risk + High dependency → `Fragility: Elevated`).
* **Recommendations**: 1–2 bullets per entity (“monitor X corridor”, “validate metering on days with imbalance%>5%”).

**Integration**

* When returning cluster results or metrics tables, enrich rows with `business_label`, `risk_badges`, `notes`.
* Include a **Business Interpretation** section in Evidence Card for clustering/metrics ops.

**Acceptance**

* Unit: labeler picks expected names for synthetic profiles.
* Integration: `agent cluster --entity-type loc` shows `business_label` and `risk_badges` columns.

**Code stub**

```python
# business.py
from __future__ import annotations
import polars as pl

def label_cluster(row: dict) -> tuple[str,list[str]]:
    # Inspect standardized features and return (label, badges)
    return ("Peaky Winter LDC", ["RampRisk:High","Reversal:Low"])
```

---

## Task 18: Confidence & Calibration Utilities

**Files:** `src/data_agent/core/confidence.py`, `tests/test_confidence.py`

**Goal:** Convert numeric diagnostics into human-readable **confidence levels** (low/medium/high) with simple calibration rules.

**Features**

* Map z-scores, sample sizes, rule violations, and p/q-values to a 0–1 score.
* Bins: 0–0.33 → low, 0.34–0.66 → medium, 0.67–1.0 → high.
* Combine multiple signals (weighted average); conservative min rule when data quality is poor.

**Acceptance**

* Unit: deterministic mappings for edge cases (tiny sample → low; strong signal, clean data → high).
* Integration: causal hypotheses show calibrated confidence.

---

## Task 19: Artifacts: Correlation Heatmaps & Causal Briefs

**Files:** `src/data_agent/core/render.py`, `tests/test_render.py`

**Goal:** Produce minimal, clean PNG artifacts without GUI requirements.

**Features**

* **Correlation heatmap**: save to `artifacts/plots/corr_{run_id}.png`; annotate top |r| pairs.
* **Causal brief**: simple Markdown file `artifacts/briefs/causal_{run_id}.md` summarizing hypotheses & evidence.

**Acceptance**

* Unit: renderer writes files and returns paths.
* Integration: relevant CLI commands print file paths in the footer.

---

## Task 20: README updates & Examples for New Features

**Files:** `README.md`, `examples/queries.md`

**Goal:** Document new capabilities with copy‑pasteable examples and sample outputs.

**Steps**

* Add section: **Causal Hypotheses** with sample CLI and redacted output.
* Add section: **Correlation Analysis** with matrix/heatmap and p/q-values.
* Add **Business Interpretation** examples with cluster labels & risk badges.

**Acceptance**

* Fresh reader can run: `agent ask "why did <pipeline> deliveries spike in <month/year>?"` and `agent ask "correlate states by sum in 2022 (pearson)"` successfully.

---














## Testing Matrix (summary)

* Unit: plan schema, ops, rules, metrics.
* Integration: deterministic templates on golden; evidence snapshot.
* E2E: CLI commands via subprocess.

---

## Nice-to-haves (Post‑MVP tickets)

* Evidence charts (matplotlib) saved to `artifacts/` and referenced in CLI.
* Node/corridor network graphs with NetworkX.
* Peer comparison cohorting and idiosyncrasy tags.
