# Agent Plans

This directory contains example DAG plans for the SynMax Data Agent. These plans demonstrate the structure and capabilities of the agent's planning system.

## Plan Structure

Each plan is a JSON file containing a directed acyclic graph (DAG) with the following structure:

```json
{
  "nodes": [
    {
      "id": "unique_step_id",
      "op": "operation_name", 
      "params": {
        "parameter": "value"
      }
    }
  ],
  "edges": [
    {
      "src": "source_step_id",
      "dst": "destination_step_id"
    }
  ],
  "inputs": ["raw"],
  "outputs": ["final_step_id"]
}
```

## Available Operations

### Data Operations
- `filter`: Apply column filters (`=`, `!=`, `>`, `<`, `>=`, `<=`, `between`, `in`, `contains`)
- `aggregate`: Group by columns and compute metrics (`sum`, `mean`, `count`, `min`, `max`)
- `rank`: Sort by columns with optional descending order
- `limit`: Limit number of rows returned
- `resample`: Time-based resampling for temporal analysis

### Analysis Operations
- `changepoint`: Detect regime shifts and significant changes in time series
- `cluster`: Entity clustering with interpretable naming
- `stl_deseasonalize`: Seasonal-trend decomposition for time series
- `evidence_collect`: Generate evidence cards for transparency

### Utility Operations
- `checkpoint`: Materialize intermediate results to disk
- `cache_read`: Read from cache if available
- `cache_write`: Write results to cache

## Example Plans

### Simple Aggregation (`simple_aggregation_example.json`)

A basic plan that aggregates pipeline data and returns the top 10 by volume:

```bash
# Execute this plan
poetry run python -m data_agent.cli run --plan examples/agent_plans/simple_aggregation_example.json
```

**Plan flow**: Raw data → Aggregate by pipeline → Rank by volume → Limit to 10 → Collect evidence

### Regime Shift Detection (`regime_shifts_2021.json`)

A complex plan that detects significant changes in gas flow patterns:

```bash
# Execute with custom settings
poetry run python -m data_agent.cli run --plan examples/agent_plans/regime_shifts_2021.json --materialize heavy --export regime_analysis.json
```

**Plan flow**: Raw data → Filter by date → Aggregate daily → Deseasonalize → Detect changepoints → Rank by magnitude → Limit results → Collect evidence

## Creating Custom Plans

### Method 1: Generate from Natural Language

```bash
# Generate a plan from a question
poetry run python -m data_agent.cli plan "detect anomalies in Michigan pipelines" --export my_plan.json --dry-run

# Review the plan
cat my_plan.json | jq '.nodes[] | {id, op, params}'

# Execute the plan
poetry run python -m data_agent.cli run --plan my_plan.json --export results.json
```

### Method 2: Manual Construction

Create a JSON file following the structure above. Key considerations:

1. **Node IDs**: Use short, descriptive IDs (`f` for filter, `a` for aggregate, etc.)
2. **Dependencies**: Ensure edges create a valid DAG with no cycles
3. **Data Flow**: Each node receives input from its dependencies via edges
4. **Evidence**: Include `evidence_collect` as the final step for transparency

### Method 3: Modify Existing Plans

```bash
# Copy an existing plan
cp examples/agent_plans/simple_aggregation_example.json my_custom_plan.json

# Edit parameters (e.g., change groupby columns, adjust limits)
# Execute modified plan
poetry run python -m data_agent.cli run --plan my_custom_plan.json
```

## Plan Execution Options

### Materialization Strategies

Control memory vs. disk usage:

```bash
# Materialize all steps (safest for large datasets)
poetry run python -m data_agent.cli run --plan my_plan.json --materialize all

# Materialize only heavy operations (default)
poetry run python -m data_agent.cli run --plan my_plan.json --materialize heavy

# Keep everything in memory (fastest)
poetry run python -m data_agent.cli run --plan my_plan.json --materialize never
```

### Caching Control

Optimize performance for repeated executions:

```bash
# Use cache with custom TTL
poetry run python -m data_agent.cli run --plan my_plan.json --cache-ttl 24

# Bypass cache for fresh results
poetry run python -m data_agent.cli run --plan my_plan.json --no-cache
```

### Export Options

Save results for further analysis:

```bash
# Auto-generate filename in artifacts/outputs/
poetry run python -m data_agent.cli run --plan my_plan.json --export auto

# Custom filename
poetry run python -m data_agent.cli run --plan my_plan.json --export my_results.json
```

## Common Plan Patterns

### Time Series Analysis
```json
{
  "nodes": [
    {"id": "f", "op": "filter", "params": {"column": "eff_gas_day", "op": "between", "value": ["2022-01-01", "2022-12-31"]}},
    {"id": "r", "op": "resample", "params": {"freq": "1M", "date_col": "eff_gas_day"}},
    {"id": "a", "op": "aggregate", "params": {"groupby": ["month"], "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]}}
  ],
  "edges": [
    {"src": "raw", "dst": "f"},
    {"src": "f", "dst": "r"},
    {"src": "r", "dst": "a"}
  ]
}
```

### Entity Clustering
```json
{
  "nodes": [
    {"id": "c", "op": "cluster", "params": {"entity_type": "counterparty", "k": 6, "features": ["dependency_score", "volume_variance"]}},
    {"id": "r", "op": "rank", "params": {"by": ["cluster_score"], "descending": true}}
  ],
  "edges": [
    {"src": "raw", "dst": "c"},
    {"src": "c", "dst": "r"}
  ]
}
```

### Anomaly Detection
```json
{
  "nodes": [
    {"id": "a", "op": "aggregate", "params": {"groupby": ["pipeline_name", "eff_gas_day"], "metrics": [{"col": "scheduled_quantity", "fn": "sum"}]}},
    {"id": "c", "op": "changepoint", "params": {"column": "sum_scheduled_quantity", "method": "pelt", "min_confidence": 0.7, "groupby": ["pipeline_name"]}}
  ],
  "edges": [
    {"src": "raw", "dst": "a"},
    {"src": "a", "dst": "c"}
  ]
}
```

## Debugging Plans

### Validation Errors

If a plan fails validation:

```bash
# Check plan structure
poetry run python -m data_agent.cli plan "test query" --dry-run --export test_plan.json

# Compare with working examples
diff test_plan.json examples/agent_plans/simple_aggregation_example.json
```

### Execution Errors

If a plan fails during execution:

1. **Enable debug logging**: `export LOG_LEVEL=DEBUG`
2. **Use dry-run mode**: Add `--dry-run` to see estimates without execution
3. **Check materialization**: Try `--materialize all` for memory issues
4. **Clear cache**: Use `--no-cache` to avoid stale data

### Performance Issues

For slow plan execution:

1. **Add checkpoints**: Include `checkpoint` operations before expensive steps
2. **Adjust materialization**: Use `heavy` or `all` strategies
3. **Optimize filters**: Move filters early in the DAG
4. **Reduce data**: Add `limit` operations where appropriate

## Best Practices

1. **Keep plans focused**: One plan per analysis question
2. **Use descriptive IDs**: Make plans readable (`filter_date` vs `f1`)
3. **Include evidence**: Always end with `evidence_collect` for transparency
4. **Test incrementally**: Build plans step by step, testing each addition
5. **Document parameters**: Add comments explaining non-obvious parameter choices
6. **Version control**: Keep plans in git for reproducibility

## Integration with Examples

These plans work seamlessly with the golden dataset:

```bash
# Load the golden dataset
poetry run python -m data_agent.cli load examples/golden.parquet

# Execute any plan
poetry run python -m data_agent.cli run --plan examples/agent_plans/simple_aggregation_example.json

# Compare with natural language equivalent
poetry run python -m data_agent.cli ask "top 10 pipelines by total volume"
```

The plans in this directory serve as both examples and templates for building your own custom analyses.
