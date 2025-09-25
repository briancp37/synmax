# Example Queries

This document provides example queries demonstrating the capabilities of the SynMax Data Agent, along with expected output formats and evidence cards.

## Basic Data Retrieval

### Simple Aggregations

**Query**: Total gas deliveries for a specific pipeline
```bash
poetry run python -m data_agent.cli ask "total deliveries for ANR Pipeline Company in 2022"
```

**Expected Output**:
```
Answer:
┌─────────────────────────┐
│ sum_scheduled_quantity  │
│ ---                     │
│ f64                     │
╞═════════════════════════╡
│ 123456789.0            │
└─────────────────────────┘

Evidence Card:
• Rows out: 1
• Columns: sum_scheduled_quantity
• Filters applied:
  - pipeline_name = ANR Pipeline Company
  - rec_del_sign = -1
  - eff_gas_day between 2022-01-01 and 2022-12-31
• Metrics:
  - sum(scheduled_quantity)
• Runtime: 23.4ms plan, 156.8ms collect
• Cache: miss
```

---

**Query**: Top states by gas volume
```bash
poetry run python -m data_agent.cli ask "top 5 states by total gas volume in 2022"
```

**Expected Output**:
```
Answer:
┌───────────┬─────────────────────────┐
│ state_abb │ sum_scheduled_quantity  │
│ ---       │ ---                     │
│ str       │ f64                     │
╞═══════════╪═════════════════════════╡
│ TX        │ 987654321.0            │
│ LA        │ 876543210.0            │
│ OK        │ 765432109.0            │
│ PA        │ 654321098.0            │
│ CO        │ 543210987.0            │
└───────────┴─────────────────────────┘

Evidence Card:
• Rows out: 5
• Columns: state_abb, sum_scheduled_quantity
• Filters applied:
  - eff_gas_day between 2022-01-01 and 2022-12-31
• Grouped by: state_abb
• Metrics:
  - sum(scheduled_quantity)
• Sorted by: sum_scheduled_quantity (desc)
• Limited to: 5 rows
• Runtime: 45.1ms plan, 234.7ms collect
```

### Time Series Queries

**Query**: Monthly gas flow trends
```bash
poetry run python -m data_agent.cli ask "monthly gas deliveries trend for Texas Eastern in 2022"
```

**Expected Output**:
```
Answer:
┌─────────────┬─────────────────────────┐
│ month       │ sum_scheduled_quantity  │
│ ---         │ ---                     │
│ str         │ f64                     │
╞═════════════╪═════════════════════════╡
│ 2022-01     │ 45678901.0             │
│ 2022-02     │ 43210987.0             │
│ 2022-03     │ 47890123.0             │
│ ...         │ ...                     │
└─────────────┴─────────────────────────┘

Evidence Card:
• Rows out: 12
• Columns: month, sum_scheduled_quantity
• Filters applied:
  - pipeline_name contains Texas Eastern
  - rec_del_sign = -1
  - eff_gas_day between 2022-01-01 and 2022-12-31
• Resample: 1M frequency on eff_gas_day
• Grouped by: month
• Metrics:
  - sum(scheduled_quantity)
• Runtime: 67.3ms plan, 189.4ms collect
```

## Advanced Analytics

### Clustering Analysis

**Query**: Cluster counterparties by dependency concentration
```bash
poetry run python -m data_agent.cli ask "cluster counterparties by their dependency concentration"
```

**Expected Output**:
```
Answer:
┌────────────────────────┬────────────┬──────────────────────┬──────────────────┐
│ counterparty_name      │ cluster_id │ cluster_name         │ dependency_score │
│ ---                    │ ---        │ ---                  │ ---              │
│ str                    │ i64        │ str                  │ f64              │
╞════════════════════════╪════════════╪══════════════════════╪══════════════════╡
│ Major Utility Corp     │ 0          │ High-Dependence      │ 0.89             │
│ Industrial Complex A   │ 0          │ High-Dependence      │ 0.85             │
│ Power Plant Network    │ 1          │ Diversified-Large    │ 0.45             │
│ Regional LDC           │ 2          │ Moderate-Dependence  │ 0.67             │
│ Small Industrial       │ 3          │ Low-Volume           │ 0.23             │
└────────────────────────┴────────────┴──────────────────────┴──────────────────┘

Evidence Card:
• Rows out: 156
• Columns: counterparty_name, cluster_id, cluster_name, dependency_score
• Operation: cluster
• Parameters:
  - entity_type: counterparty
  - k: 6
  - features: dependency_concentration, volume_variance, seasonal_pattern
• Entities Clustered: 156
• Cluster Size Range: 18-35
• Silhouette Score: 0.67
• Runtime: 234.5ms plan, 1456.2ms collect
```

### Anomaly Detection

**Query**: Detect flow anomalies and changepoints
```bash
poetry run python -m data_agent.cli ask "detect significant flow changes in Michigan pipelines since June 2022"
```

**Expected Output**:
```
Answer:
┌──────────────┬─────────────────────┬─────────────┬────────────┬──────────────┐
│ eff_gas_day  │ pipeline_name       │ location    │ confidence │ change_type  │
│ ---          │ ---                 │ ---         │ ---        │ ---          │
│ date         │ str                 │ str         │ f64        │ str          │
╞══════════════╪═════════════════════╪═════════════╪════════════╪══════════════╡
│ 2022-07-15   │ ANR Pipeline        │ Compressor  │ 0.89       │ volume_spike │
│ 2022-08-03   │ Consumers Energy    │ Storage     │ 0.76       │ trend_change │
│ 2022-09-12   │ Michigan Gas        │ Delivery    │ 0.82       │ volatility   │
└──────────────┴─────────────────────┴─────────────┴────────────┴──────────────┘

Evidence Card:
• Rows out: 3
• Columns: eff_gas_day, pipeline_name, location, confidence, change_type
• Operation: changepoint
• Parameters:
  - min_confidence: 0.7
  - penalty: 3.0
  - model: rbf
• Filters applied:
  - state_abb = MI
  - eff_gas_day >= 2022-06-01
• Changepoints detected: 3 of 15 candidates
• Runtime: 156.7ms plan, 2341.9ms collect
```

### Pattern Recognition

**Query**: Identify seasonal patterns
```bash
poetry run python -m data_agent.cli ask "identify seasonal patterns in Texas gas production"
```

**Expected Output**:
```
Answer:
┌─────────┬─────────────────────────┬──────────────────┬─────────────────┐
│ month   │ avg_scheduled_quantity  │ seasonal_factor  │ pattern_type    │
│ ---     │ ---                     │ ---              │ ---             │
│ i32     │ f64                     │ f64              │ str             │
╞═════════╪═════════════════════════╪══════════════════╪═════════════════╡
│ 1       │ 2456789.0              │ 1.23             │ winter_peak     │
│ 2       │ 2345678.0              │ 1.18             │ winter_high     │
│ 3       │ 2123456.0              │ 1.07             │ spring_moderate │
│ ...     │ ...                     │ ...              │ ...             │
│ 12      │ 2398765.0              │ 1.21             │ winter_peak     │
└─────────┴─────────────────────────┴──────────────────┴─────────────────┘

Evidence Card:
• Rows out: 12
• Columns: month, avg_scheduled_quantity, seasonal_factor, pattern_type
• Operation: metric_compute
• Filters applied:
  - state_abb = TX
  - rec_del_sign = 1
  - category_short = Production
• Grouped by: month
• Metrics:
  - avg(scheduled_quantity)
• Seasonal decomposition applied
• Runtime: 89.2ms plan, 445.6ms collect
```

## Data Quality & Rules

### Rule Violations

**Query**: Check data quality rules
```bash
poetry run python -m data_agent.cli rules --pipeline "ANR Pipeline Company" --since 2022-06-01
```

**Expected Output**:
```
Data Quality Rules Summary
┌─────────────────┬──────────────────────────────────────┬────────────┬──────────────┐
│ Rule ID         │ Description                          │ Violations │ Sample Count │
│ ---             │ ---                                  │ ---        │ ---          │
│ str             │ str                                  │ i64        │ i64          │
╞═════════════════╪══════════════════════════════════════╪════════════╪══════════════╡
│ volume_balance  │ Receipt/delivery balance check       │ 12         │ 1000         │
│ negative_flows  │ Unexpected negative scheduled qty    │ 3          │ 1000         │
│ missing_dates   │ Missing effective gas day entries    │ 0          │ 1000         │
│ duplicate_keys  │ Duplicate pipeline-location-date     │ 1          │ 1000         │
└─────────────────┴──────────────────────────────────────┴────────────┴──────────────┘

Summary: 4 rules checked, 16 total violations found across 1000 records
```

### Event Detection

**Query**: Detect significant events
```bash
poetry run python -m data_agent.cli events --pipeline "Texas Eastern" --top 5 --export auto
```

**Expected Output**:
```
Significant Events Detected
┌──────────────┬─────────────────┬────────────┬─────────────────┬──────────────────┐
│ eff_gas_day  │ location_name   │ confidence │ magnitude       │ event_type       │
│ ---          │ ---             │ ---        │ ---             │ ---              │
│ date         │ str             │ f64        │ f64             │ str              │
╞══════════════╪═════════════════╪════════════╪═════════════════╪══════════════════╡
│ 2022-02-14   │ Compressor 1A   │ 0.94       │ 234567.0        │ winter_surge     │
│ 2022-07-21   │ Storage Hub     │ 0.87       │ -187654.0       │ maintenance_drop │
│ 2022-11-03   │ Delivery Point  │ 0.82       │ 298765.0        │ demand_spike     │
│ 2022-08-15   │ Receipt Point   │ 0.79       │ -156789.0       │ supply_disruption│
│ 2022-12-25   │ Main Line       │ 0.75       │ -98765.0        │ holiday_reduction│
└──────────────┴─────────────────┴────────────┴─────────────────┴──────────────────┘

Summary: Detected 5 significant events with potential operational impact
Results exported to artifacts/outputs/20241224_143052_a7b8c9d2.json
```

## Comparative Analysis

### Pipeline Comparison

**Query**: Compare pipeline performance metrics
```bash
poetry run python -m data_agent.cli ask "compare utilization rates between ANR and Texas Eastern pipelines in 2022"
```

**Expected Output**:
```
Answer:
┌─────────────────────┬─────────────────┬─────────────────┬──────────────────┐
│ pipeline_name       │ utilization_pct │ avg_daily_flow  │ reliability_score│
│ ---                 │ ---             │ ---             │ ---              │
│ str                 │ f64             │ f64             │ f64              │
╞═════════════════════╪═════════════════╪═════════════════╪══════════════════╡
│ ANR Pipeline Co     │ 78.5            │ 1234567.0       │ 0.94             │
│ Texas Eastern       │ 82.3            │ 1456789.0       │ 0.91             │
└─────────────────────┴─────────────────┴─────────────────┴──────────────────┘

Evidence Card:
• Rows out: 2
• Columns: pipeline_name, utilization_pct, avg_daily_flow, reliability_score
• Operation: metric_compute
• Filters applied:
  - pipeline_name in [ANR Pipeline Company, Texas Eastern Transmission]
  - eff_gas_day between 2022-01-01 and 2022-12-31
• Grouped by: pipeline_name
• Metrics:
  - utilization_pct: custom calculation
  - avg(scheduled_quantity)
  - reliability_score: variance-based metric
• Runtime: 123.4ms plan, 567.8ms collect
```

### Regional Analysis

**Query**: Regional flow pattern analysis
```bash
poetry run python -m data_agent.cli ask "analyze regional gas flow imbalances in the Gulf Coast region"
```

**Expected Output**:
```
Answer:
┌───────────┬─────────────┬──────────────┬─────────────────┬──────────────────┐
│ state_abb │ net_receipts│ net_deliveries│ imbalance_pct   │ risk_category    │
│ ---       │ ---         │ ---           │ ---             │ ---              │
│ str       │ f64         │ f64           │ f64             │ str              │
╞═══════════╪═════════════╪══════════════╪═════════════════╪══════════════════╡
│ TX        │ 9876543.0   │ 8765432.0     │ 12.7            │ moderate_surplus │
│ LA        │ 8765432.0   │ 9876543.0     │ -11.2           │ moderate_deficit │
│ MS        │ 2345678.0   │ 2456789.0     │ -4.5            │ balanced         │
│ AL        │ 3456789.0   │ 3234567.0     │ 6.9             │ slight_surplus   │
└───────────┴─────────────┴──────────────┴─────────────────┴──────────────────┘

Evidence Card:
• Rows out: 4
• Columns: state_abb, net_receipts, net_deliveries, imbalance_pct, risk_category
• Operation: metric_compute
• Filters applied:
  - state_abb in [TX, LA, MS, AL]
• Grouped by: state_abb
• Metrics:
  - sum(receipts), sum(deliveries)
  - imbalance_pct: (receipts-deliveries)/total * 100
• Risk categorization applied
• Runtime: 78.9ms plan, 234.1ms collect
```

## Export Examples

### Basic Export

**Query**: Export query results to JSON
```bash
poetry run python -m data_agent.cli ask "top 10 delivery locations by volume" --export auto
```

**Output**: 
```
Results exported to: artifacts/outputs/20241224_143052_a7b8c9d2.json
```

**Export File Structure**:
```json
{
  "run_id": "20241224_143052_a7b8c9d2",
  "timestamp": "2024-12-24T14:30:52.123456",
  "question": "top 10 delivery locations by volume",
  "plan": {
    "filters": [
      {"column": "rec_del_sign", "op": "=", "value": -1}
    ],
    "aggregate": {
      "groupby": ["loc_name"],
      "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]
    },
    "sort": {
      "by": ["sum_scheduled_quantity"],
      "desc": true,
      "limit": 10
    }
  },
  "answer": {
    "table": [
      {"loc_name": "Houston Hub", "sum_scheduled_quantity": 9876543.0},
      {"loc_name": "Chicago Gate", "sum_scheduled_quantity": 8765432.0}
    ],
    "evidence": {
      "rows_out": 10,
      "columns": ["loc_name", "sum_scheduled_quantity"],
      "filters": [{"column": "rec_del_sign", "op": "=", "value": -1}],
      "aggregate": {"groupby": ["loc_name"], "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]},
      "sort": {"by": ["sum_scheduled_quantity"], "desc": true, "limit": 10},
      "timings_ms": {"plan": 45.2, "collect": 178.9},
      "cache": {"hit": false}
    }
  },
  "metadata": {
    "rows_returned": 10,
    "columns": ["loc_name", "sum_scheduled_quantity"],
    "plan_type": "basic",
    "cache_hit": false,
    "execution_time_ms": {
      "plan": 45.2,
      "collect": 178.9,
      "total": 224.1
    }
  }
}
```

## Performance Examples

### Cache Performance

**First Query** (cache miss):
```bash
poetry run python -m data_agent.cli ask "monthly trends for Texas pipelines"
# Runtime: 45.2ms plan, 1234.5ms collect
# Cache: miss
```

**Repeated Query** (cache hit):
```bash
poetry run python -m data_agent.cli ask "monthly trends for Texas pipelines"
# Runtime: 12.3ms plan, 23.4ms collect
# Cache: hit
```

### Large Dataset Performance

**Query**: Analysis on full dataset
```bash
poetry run python -m data_agent.cli ask "cluster all counterparties by flow patterns" --no-cache
```

**Expected Performance**:
```
Evidence Card:
• Entities Clustered: 2,847
• Features: 12
• Silhouette Score: 0.73
• Runtime: 456.7ms plan, 12,345.6ms collect
• Memory usage: ~1.2GB peak
```

## Error Handling Examples

### Invalid Query
```bash
poetry run python -m data_agent.cli ask "nonexistent column analysis"
```

**Output**:
```
Error: Column 'nonexistent_column' not found in dataset
Available columns: pipeline_name, state_abb, eff_gas_day, ...
```

### API Key Missing
```bash
# With no API keys set
poetry run python -m data_agent.cli ask "complex pattern analysis" --planner llm
```

**Output**:
```
Error: No LLM API keys configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY
Falling back to deterministic planner for supported query patterns.
```

### Dataset Not Loaded
```bash
poetry run python -m data_agent.cli ask "any query"
```

**Output**:
```
Error: Dataset not found. Please run 'agent load --auto' first to download the dataset.
```

## Tips for Best Results

### Query Formulation

**Good queries**:
- "total deliveries for ANR Pipeline in January 2022"
- "cluster counterparties by dependency concentration"
- "detect flow anomalies in Texas since June"

**Avoid**:
- Overly vague: "show me data"
- Multiple questions: "show deliveries and also cluster entities"
- Impossible requests: "predict future prices"

### Performance Optimization

1. **Use caching**: Avoid `--no-cache` unless necessary
2. **Be specific**: Filter by date ranges, pipelines, or regions
3. **Export large results**: Use `--export auto` for complex analyses
4. **Monitor resources**: Large clustering operations use significant memory

### Troubleshooting

1. **Enable debug logging**: `export LOG_LEVEL=DEBUG`
2. **Check cache**: `poetry run python -m data_agent.cli cache --stats`
3. **Dry run first**: Use `--dry-run` to validate query plans
4. **Clear cache**: If getting stale results, clear cache

---

These examples demonstrate the full range of capabilities available in the SynMax Data Agent. Each query type provides rich evidence cards explaining the methodology and parameters used, ensuring transparency and reproducibility in all analyses.
