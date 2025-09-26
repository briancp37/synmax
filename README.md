# SynMax Data Agent

A **chat-based agent in Python** that can interact with gas pipeline datasets and answer user questions through natural language queries. The agent handles everything from simple data retrieval to advanced analysis including pattern recognition, anomaly detection, and causal hypothesis generation.

## Overview

The Data Agent provides:

- **Pattern recognition** (clustering, correlations, trends)
- **Anomaly detection** (outliers, rule violations, changepoint detection)  
- **Causal hypothesis generation** with evidence and limitations
- **Natural language query interface** via CLI
- **Comprehensive evidence tracking** for all analyses
- **Caching system** for improved performance
- **Export capabilities** for results and evidence

## Features

### Core Capabilities

- **Data Ingestion**: Automatically infers schema & types, handles missing values
- **Natural Language Understanding**: Converts questions into executable analysis plans
- **Query Planning & Execution**: Optimized execution over large datasets using Polars/DuckDB
- **Evidence Generation**: Returns concise answers plus supporting evidence (methods used, selected columns, filters)
- **LLM Answer Generation**: Automatically generates prose summaries of query results using OpenAI/Anthropic models
- **Deterministic & Analytic Queries**: Handles both simple counts and complex pattern analysis

### Analysis Types

- **Metric Computation**: Aggregations, rollups, time series analysis
- **Clustering**: Entity clustering with interpretable cluster naming
- **Changepoint Detection**: Identifies significant events in time series data
- **Rules Engine**: Data quality validation and violation detection
- **Pattern Recognition**: Correlation analysis and trend detection

## Installation & Quick Start

### Prerequisites

- Python 3.10+
- Poetry (recommended) or pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd synmax
   ```

2. **Install dependencies with Poetry**:
   ```bash
   poetry install
   ```

   Or with pip:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Environment Variables

The agent supports both OpenAI and Anthropic LLMs. Set at least one:

```bash
# Required: Set at least one API key
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Logging configuration
LOG_LEVEL=INFO
```

### Dataset Setup

The agent requires a gas pipeline dataset. You have two options:

#### Option 1: Auto-download (Recommended)
```bash
poetry run python -m data_agent.cli load --auto
```

This downloads the dataset from the provided Google Drive link and stores it in `./data/pipeline_data.parquet`.

#### Option 2: Local file
If you have the dataset file locally:
```bash
poetry run python -m data_agent.cli load --path /path/to/your/dataset.parquet
```

### Quick Start

1. **Load the dataset**:
   ```bash
   poetry run python -m data_agent.cli load --auto
   ```

2. **Ask your first question**:
   ```bash
   poetry run python -m data_agent.cli ask "What are the top 5 pipelines by total volume in 2022?"
   ```

3. **Try advanced analysis**:
   ```bash
   poetry run python -m data_agent.cli ask "Detect anomalies in Texas gas flow patterns"
   ```

## Usage

### Basic Commands

The agent provides several CLI commands:

#### `load` - Load Dataset
```bash
# Auto-download dataset
poetry run python -m data_agent.cli load --auto

# Load from local path
poetry run python -m data_agent.cli load --path data/pipeline_data.parquet
```

#### `ask` - Natural Language Queries
```bash
# Basic query
poetry run python -m data_agent.cli ask "total deliveries for ANR Pipeline in January 2022"

# With specific planner
poetry run python -m data_agent.cli ask "cluster counterparties by dependency" --planner llm

# Export results
poetry run python -m data_agent.cli ask "top states by gas volume" --export auto

# Dry run (show plan without execution)
poetry run python -m data_agent.cli ask "monthly trends" --dry-run

# Suppress LLM-generated answer (show only data)
poetry run python -m data_agent.cli ask "top 5 pipelines" --no-answer

# Bypass cache
poetry run python -m data_agent.cli ask "recent anomalies" --no-cache
```

#### `rules` - Data Quality Analysis
```bash
# Run all data quality rules
poetry run python -m data_agent.cli rules

# Filter by pipeline
poetry run python -m data_agent.cli rules --pipeline "ANR Pipeline Company"

# Show violations since specific date
poetry run python -m data_agent.cli rules --since 2022-06-01
```

#### `events` - Changepoint Detection
```bash
# Detect events across all pipelines
poetry run python -m data_agent.cli events

# Focus on specific pipeline
poetry run python -m data_agent.cli events --pipeline "Texas Eastern"

# Adjust sensitivity
poetry run python -m data_agent.cli events --min-confidence 0.5

# Export results
poetry run python -m data_agent.cli events --export events_analysis.json
```

#### `cluster` - Entity Clustering
```bash
# Cluster locations
poetry run python -m data_agent.cli cluster --entity-type loc --k 6

# Cluster counterparties
poetry run python -m data_agent.cli cluster --entity-type counterparty --k 8

# Export clustering results
poetry run python -m data_agent.cli cluster --entity-type loc --export clusters.json
```

#### `cache` - Cache Management
```bash
# Show cache statistics
poetry run python -m data_agent.cli cache --stats

# Clear cache
poetry run python -m data_agent.cli cache --clear
```

### Export Functionality

The agent supports exporting results in structured JSON format:

```bash
# Auto-generate filename in artifacts/outputs/
poetry run python -m data_agent.cli ask "your question" --export auto

# Custom export path
poetry run python -m data_agent.cli ask "your question" --export results.json
```

Exported files include:
- Original question and generated plan
- Complete results table
- Evidence card with methodology details
- Execution metadata (timing, cache status, etc.)

## Example Queries & Outputs

### Simple Aggregation
```bash
poetry run python -m data_agent.cli ask "total gas deliveries in Texas for 2022"
```

**Output**:
```
Plan hash: abc123...
Steps executed: 3
Final result: 1 rows, 1 columns

Evidence:
  Plan executed in 3 steps
  Detailed evidence: artifacts/outputs/abc123...json

Results:
┌─────────────────────┐
│ sum_scheduled_qty   │
│ ---                 │
│ f64                 │
╞═════════════════════╡
│ 1234567890.0       │
└─────────────────────┘

Answer:
The analysis shows total gas deliveries in Texas for 2022 reached approximately 1.23 billion units. This aggregation was performed across all pipeline data filtered by state (TX) and the specified date range (2022-01-01 to 2022-12-31).
```

### Pattern Recognition
```bash
poetry run python -m data_agent.cli ask "cluster counterparties by their dependency concentration"
```

**Output**:
```
Plan hash: def456...
Steps executed: 5
Final result: 2 rows, 4 columns

Evidence:
  Plan executed in 5 steps
  Detailed evidence: artifacts/outputs/def456...json

Results:
┌─────────────────┬─────────────┬──────────────────┬─────────────────┐
│ counterparty    │ cluster_id  │ cluster_name     │ concentration   │
│ ---             │ ---         │ ---              │ ---             │
│ str             │ i64         │ str              │ f64             │
╞═════════════════╪═════════════╪══════════════════╪═════════════════╡
│ Utility Corp A  │ 0          │ High-Dependence  │ 0.85           │
│ Power Plant B   │ 1          │ Diversified      │ 0.32           │
└─────────────────┴─────────────┴──────────────────┴─────────────────┘

Answer:
The clustering analysis identified two distinct counterparty groups based on dependency concentration. Utility Corp A shows high dependence with 85% concentration, while Power Plant B demonstrates a diversified approach with only 32% concentration. The analysis achieved a silhouette score of 0.67, indicating well-separated clusters.
```

### Anomaly Detection
```bash
poetry run python -m data_agent.cli ask "detect flow anomalies in Michigan pipelines since June 2022"
```

**Output**:
```
Answer:
┌──────────────┬─────────────┬──────────────┬─────────────┐
│ eff_gas_day  │ pipeline    │ anomaly_type │ confidence  │
│ ---          │ ---         │ ---          │ ---         │
│ date         │ str         │ str          │ f64         │
╞══════════════╪═════════════╪══════════════╪═════════════╡
│ 2022-07-15   │ ANR        │ volume_spike  │ 0.89        │
│ 2022-08-03   │ Consumers  │ flow_reversal │ 0.76        │
└──────────────┴─────────────┴──────────────┴─────────────┘

Evidence Card:
• Operation: changepoint
• Parameters:
  - min_confidence: 0.7
  - penalty: 3.0
• Filters applied:
  - state_abb = MI
  - eff_gas_day >= 2022-06-01
• Changepoints detected: 2
• Runtime: 156.7ms plan, 2341.9ms collect
```

## Architecture & Design

### Core Components

- **CLI Interface** (`cli.py`): Typer-based command-line interface
- **Query Planner** (`core/planner.py`): Converts natural language to execution plans
- **Executor** (`core/executor.py`): Executes plans against data with evidence tracking
- **Cache System** (`cache/`): Redis-like caching for query results
- **Rules Engine** (`rules/`): Data quality validation framework
- **Analysis Modules**:
  - `core/metrics.py`: Statistical computations
  - `core/cluster.py`: Entity clustering with interpretable naming
  - `core/events.py`: Changepoint detection

### Data Processing

- **Polars**: Primary data processing engine (lazy evaluation)
- **DuckDB**: Heavy aggregations and complex queries
- **Schema Inference**: Automatic type detection and validation
- **Missing Value Handling**: Configurable strategies per column type

### LLM Integration

- **Dual Provider Support**: OpenAI GPT-4 and Anthropic Claude
- **Structured Output**: JSON schema validation for plans
- **Fallback Logic**: Deterministic patterns when LLM unavailable
- **Cost Optimization**: Caching and plan reuse

## Development

### Code Quality

The project maintains high code quality standards:

```bash
# Format code
make format

# Lint code  
make lint

# Type checking
make typecheck

# Run tests
make test

# All checks (required for PRs)
make format lint typecheck test
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=data_agent

# Run specific test categories
poetry run pytest tests/test_planner.py
poetry run pytest tests/test_executor.py
poetry run pytest tests/test_cli.py
```

### Project Structure

```
synmax/
├── src/data_agent/          # Main package
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── cache/              # Caching system
│   ├── core/               # Core analysis engine
│   │   ├── planner.py      # Query planning
│   │   ├── executor.py     # Plan execution
│   │   ├── evidence.py     # Evidence generation
│   │   ├── metrics.py      # Statistical functions
│   │   ├── cluster.py      # Clustering analysis
│   │   ├── events.py       # Changepoint detection
│   │   └── export.py       # Result export
│   ├── ingest/             # Data loading and preprocessing
│   ├── rules/              # Data quality rules
│   └── utils/              # Utilities
├── tests/                  # Test suite
├── examples/               # Example data and queries
├── artifacts/              # Generated outputs (gitignored)
└── data/                   # Dataset storage (gitignored)
```

## Assumptions & Limitations

### Data Assumptions

- **Schema Stability**: Pipeline data follows consistent schema across time periods
- **Date Formats**: Effective gas day in YYYY-MM-DD format
- **Numeric Precision**: Sufficient precision for volume calculations
- **Missing Values**: Handled as nulls, not zeros (important for flow calculations)

### Analysis Limitations

- **Temporal Scope**: Analysis quality depends on data time range coverage
- **Clustering Stability**: Results may vary with different random seeds (mitigated with fixed seeds)
- **Changepoint Sensitivity**: Detection thresholds may need tuning per pipeline
- **LLM Dependency**: Advanced queries require API access; fallback to deterministic patterns

### Performance Considerations

- **Memory Usage**: Large datasets processed lazily but final results materialized
- **Cache Dependencies**: Performance gains depend on query similarity
- **Network Latency**: LLM calls add 1-3 second latency per query
- **Disk I/O**: Dataset loading time proportional to file size

### Accuracy Considerations

- **Statistical Methods**: Use established algorithms (scikit-learn, ruptures) with documented limitations
- **Hypothesis Generation**: Causal claims marked as hypotheses with evidence levels
- **Data Quality**: Rules engine identifies but doesn't automatically correct issues
- **Confidence Intervals**: Provided where statistically meaningful

## Troubleshooting

### Common Issues

**Dataset Not Found**:
```bash
# Ensure dataset is loaded
poetry run python -m data_agent.cli load --auto
```

**API Key Errors**:
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

**Memory Issues**:
```bash
# Clear cache to free memory
poetry run python -m data_agent.cli cache --clear
```

**Import Errors**:
```bash
# Reinstall in development mode
poetry install
```

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
poetry run python -m data_agent.cli ask "your question"
```

Check cache status:
```bash
poetry run python -m data_agent.cli cache --stats
```

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** ensuring all tests pass: `make format lint typecheck test`
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open Pull Request**

### Code Style

- **Python 3.10+** features encouraged
- **Type hints** required for all functions
- **Docstrings** for public APIs
- **Error handling** with informative messages
- **Logging** for debugging and monitoring

## License

This project is developed for SynMax evaluation purposes.

## Support

For questions or issues:

1. Check this README for common solutions
2. Review the `examples/` directory for usage patterns
3. Enable debug logging for detailed error information
4. Check the test suite for expected behavior examples

---

**Built with**: Python 3.10+, Polars, DuckDB, Typer, OpenAI/Anthropic APIs
