"""Tests for CLI interface."""

from typer.testing import CliRunner

from data_agent.cli import app

runner = CliRunner()


def test_cli_help():
    """Test that CLI help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "SynMax Data Agent" in result.stdout
    assert "Natural language queries over gas pipeline data" in result.stdout


def test_load_command_help():
    """Test load command help."""
    result = runner.invoke(app, ["load", "--help"])
    assert result.exit_code == 0
    assert "Load dataset from --path or auto-download to ./data." in result.stdout


def test_load_command_no_args():
    """Test load command with no arguments fails gracefully."""
    result = runner.invoke(app, ["load"])
    assert result.exit_code == 1
    assert "not found" in result.stderr


def test_load_command_with_path():
    """Test load command with non-existent path."""
    result = runner.invoke(app, ["load", "--path", "test.parquet"])
    assert result.exit_code == 1
    assert "not found" in result.stderr


def test_load_command_with_auto():
    """Test load command with auto flag and non-existent file."""
    result = runner.invoke(app, ["load", "--auto"])
    assert result.exit_code == 1
    assert "auto-download not implemented" in result.stderr


def test_load_command_success():
    """Test load command with existing golden.parquet file."""
    result = runner.invoke(app, ["load", "--path", "examples/golden.parquet"])
    assert result.exit_code == 0
    assert "Loaded dataset from: examples/golden.parquet" in result.stdout
    assert "Rows: 1,000" in result.stdout
    assert "Data dictionary written to:" in result.stdout


def test_ask_command():
    """Test ask command."""
    result = runner.invoke(app, ["ask", "What is the total flow for Texas?"])
    assert result.exit_code == 0
    assert "Question: What is the total flow for Texas?" in result.stdout
    assert "Using planner: deterministic" in result.stdout
    assert "Answer:" in result.stdout
    assert "Evidence Card:" in result.stdout


def test_ask_command_with_planner():
    """Test ask command with custom planner."""
    result = runner.invoke(app, ["ask", "test question", "--planner", "llm"])
    assert result.exit_code == 1  # Should fail because LLM planner is not implemented
    assert "Using planner: llm" in result.stdout
    assert "LLM-based planning not yet implemented" in result.stderr


def test_ask_command_with_export():
    """Test ask command with export option."""
    result = runner.invoke(app, ["ask", "test question", "--export", "output.json"])
    assert result.exit_code == 0
    assert "Results exported to: output.json" in result.stdout


def test_rules_command():
    """Test rules command."""
    result = runner.invoke(app, ["rules"])
    assert result.exit_code == 0
    assert "Running data quality rules" in result.stdout
    assert "Rules scan (placeholder)" in result.stdout


def test_rules_command_with_options():
    """Test rules command with pipeline and since options."""
    result = runner.invoke(app, ["rules", "--pipeline", "ANR", "--since", "2022-01-01"])
    assert result.exit_code == 0
    assert "Filtering by pipeline: ANR" in result.stdout
    assert "Since date: 2022-01-01" in result.stdout


def test_metrics_command_valid():
    """Test metrics command with valid metric name."""
    result = runner.invoke(app, ["metrics", "--name", "ramp_risk"])
    assert result.exit_code == 0
    assert "Computing metric: ramp_risk" in result.stdout
    assert "Metric ramp_risk (placeholder)" in result.stdout


def test_metrics_command_invalid():
    """Test metrics command with invalid metric name."""
    result = runner.invoke(app, ["metrics", "--name", "invalid_metric"])
    assert result.exit_code == 1
    assert "Invalid metric" in result.stdout


def test_events_command():
    """Test events command."""
    result = runner.invoke(app, ["events"])
    assert result.exit_code == 0
    assert "Detecting change-point events" in result.stdout
    assert "Showing top 10 events" in result.stdout


def test_events_command_with_options():
    """Test events command with options."""
    result = runner.invoke(
        app, ["events", "--pipeline", "ANR", "--since", "2022-01-01", "--top", "5"]
    )
    assert result.exit_code == 0
    assert "Pipeline: ANR" in result.stdout
    assert "Since: 2022-01-01" in result.stdout
    assert "Showing top 5 events" in result.stdout


def test_cluster_command_valid():
    """Test cluster command with valid entity type."""
    result = runner.invoke(app, ["cluster", "--entity-type", "loc", "--k", "8"])
    assert result.exit_code == 0
    assert "Clustering loc entities into 8 clusters" in result.stdout
    assert "Clustered loc entities (placeholder)" in result.stdout


def test_cluster_command_invalid():
    """Test cluster command with invalid entity type."""
    result = runner.invoke(app, ["cluster", "--entity-type", "invalid"])
    assert result.exit_code == 1
    assert "Invalid entity type" in result.stdout


def test_cache_command_clear():
    """Test cache clear command."""
    result = runner.invoke(app, ["cache", "--clear"])
    assert result.exit_code == 0
    assert "Cache cleared" in result.stdout


def test_cache_command_stats():
    """Test cache stats command."""
    result = runner.invoke(app, ["cache", "--stats"])
    assert result.exit_code == 0
    assert "Cache statistics (placeholder)" in result.stdout


def test_cache_command_conflicting_options():
    """Test cache command with conflicting options."""
    result = runner.invoke(app, ["cache", "--clear", "--stats"])
    assert result.exit_code == 1
    assert "Cannot use --clear and --stats together" in result.stdout


def test_cache_command_no_options():
    """Test cache command with no options."""
    result = runner.invoke(app, ["cache"])
    assert result.exit_code == 0
    assert "Use --clear to clear cache or --stats to show statistics" in result.stdout
