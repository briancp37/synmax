"""Tests for CLI agent commands: plan, run, and ask with new flags."""

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_agent.cli import app

runner = CliRunner()


def test_plan_command_help():
    """Test plan command help shows all options."""
    result = runner.invoke(app, ["plan", "--help"])
    assert result.exit_code == 0
    assert "Natural language question to plan" in result.stdout
    assert "--model" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--export" in result.stdout
    assert "--fallback" in result.stdout


def test_plan_command_basic():
    """Test basic plan command functionality."""
    result = runner.invoke(app, ["plan", "total receipts for ANR"])
    assert result.exit_code == 0
    assert "Generated plan for: total receipts for ANR" in result.stdout
    assert "Plan hash:" in result.stdout
    assert "Steps:" in result.stdout
    assert "Edges:" in result.stdout


def test_plan_command_with_dry_run():
    """Test plan command with dry-run flag."""
    result = runner.invoke(app, ["plan", "total receipts for ANR", "--dry-run"])
    assert result.exit_code == 0
    assert "Plan Structure:" in result.stdout
    assert "Topological order:" in result.stdout
    assert "Estimated time:" in result.stdout
    assert "Estimated memory:" in result.stdout
    assert "Steps:" in result.stdout
    assert "Edges:" in result.stdout


def test_plan_command_with_export():
    """Test plan command with export option."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        export_path = f.name

    try:
        result = runner.invoke(app, ["plan", "total receipts for ANR", "--export", export_path])
        assert result.exit_code == 0
        assert f"Plan exported to: {export_path}" in result.stdout

        # Verify the exported file exists and contains valid JSON
        assert Path(export_path).exists()
        with open(export_path) as f:
            plan_data = json.load(f)
        assert "nodes" in plan_data
        assert "edges" in plan_data
    finally:
        Path(export_path).unlink(missing_ok=True)


def test_plan_command_with_model():
    """Test plan command with model specification."""
    result = runner.invoke(app, ["plan", "total receipts for ANR", "--model", "gpt-4.1"])
    assert result.exit_code == 0
    assert "Generated plan for: total receipts for ANR" in result.stdout


def test_run_command_help():
    """Test run command help shows all options."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--plan" in result.stdout
    assert "--export" in result.stdout
    assert "--materialize" in result.stdout
    assert "--cache-ttl" in result.stdout
    assert "--cache-max-gb" in result.stdout
    assert "--no-cache" in result.stdout


def test_run_command_with_invalid_plan():
    """Test run command with non-existent plan file."""
    result = runner.invoke(app, ["run", "--plan", "nonexistent.json"])
    assert result.exit_code == 1
    assert "Plan file not found" in result.stderr


def test_run_command_with_invalid_materialize():
    """Test run command with invalid materialize option."""
    result = runner.invoke(app, ["run", "--plan", "test.json", "--materialize", "invalid"])
    assert result.exit_code == 1
    assert "Invalid materialize option: invalid" in result.stderr


def test_ask_command_help():
    """Test ask command help shows all new options."""
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--materialize" in result.stdout
    assert "--cache-ttl" in result.stdout
    assert "--cache-max-gb" in result.stdout
    assert "--no-cache" in result.stdout


def test_ask_command_with_materialize_options():
    """Test ask command with different materialize options."""
    # Test valid options
    for materialize_option in ["all", "heavy", "never"]:
        result = runner.invoke(
            app,
            [
                "ask",
                "total receipts for ANR",
                "--materialize",
                materialize_option,
                "--dry-run",  # Use dry-run to avoid actual execution
            ],
        )
        assert result.exit_code == 0, f"Failed with materialize option: {materialize_option}"


def test_ask_command_with_invalid_materialize():
    """Test ask command with invalid materialize option."""
    result = runner.invoke(app, ["ask", "total receipts for ANR", "--materialize", "invalid"])
    assert result.exit_code == 1
    assert "Invalid materialize option: invalid" in result.stderr


def test_ask_command_with_cache_options():
    """Test ask command with cache options."""
    result = runner.invoke(
        app,
        [
            "ask",
            "total receipts for ANR",
            "--cache-ttl",
            "48",
            "--cache-max-gb",
            "5.0",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Plan Structure:" in result.stdout


def test_ask_command_with_no_cache():
    """Test ask command with no-cache flag."""
    result = runner.invoke(app, ["ask", "total receipts for ANR", "--no-cache", "--dry-run"])
    assert result.exit_code == 0
    assert "Plan Structure:" in result.stdout


def test_ask_command_with_auto_export():
    """Test ask command with auto export."""
    result = runner.invoke(app, ["ask", "total receipts for ANR", "--export", "auto", "--dry-run"])
    assert result.exit_code == 0
    assert "Plan Structure:" in result.stdout


def test_ask_command_with_model_and_dry_run():
    """Test ask command with model specification and dry-run."""
    result = runner.invoke(
        app, ["ask", "total receipts for ANR", "--model", "gpt-4.1", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Plan Structure:" in result.stdout
    assert "Plan JSON:" in result.stdout


def test_ask_command_agent_mode_with_fallback():
    """Test ask command in agent mode with fallback enabled."""
    result = runner.invoke(
        app, ["ask", "total receipts for ANR", "--planner", "agent", "--fallback", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Using planner: agent" in result.stdout


def test_ask_command_agent_mode_no_fallback():
    """Test ask command in agent mode with fallback disabled."""
    result = runner.invoke(
        app, ["ask", "total receipts for ANR", "--planner", "agent", "--no-fallback", "--dry-run"]
    )
    # When fallback is disabled and no API keys are available, it should fail
    # This is expected behavior
    if result.exit_code != 0:
        assert "LLM planning failed" in result.stdout or "Error:" in result.stderr
    else:
        assert "Using planner: agent" in result.stdout


def test_plan_run_integration():
    """Test integration of plan and run commands."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        plan_path = f.name

    try:
        # First, create a plan
        result = runner.invoke(app, ["plan", "total receipts for ANR", "--export", plan_path])
        assert result.exit_code == 0
        assert Path(plan_path).exists()

        # Then try to run it (this will likely fail due to missing dataset, but should validate the plan)
        result = runner.invoke(
            app,
            [
                "run",
                "--plan",
                plan_path,
                "--materialize",
                "never",  # Use never to avoid heavy computation
            ],
        )
        # The command might fail due to missing dataset, but it should at least validate the plan
        # We check that it doesn't fail due to plan validation issues
        if result.exit_code != 0:
            assert "Plan validation failed" not in result.stderr
            assert "Invalid materialize option" not in result.stderr

    finally:
        Path(plan_path).unlink(missing_ok=True)


def test_cache_controls_validation():
    """Test validation of cache control parameters."""
    # Test negative cache TTL
    result = runner.invoke(app, ["ask", "total receipts for ANR", "--cache-ttl", "-1", "--dry-run"])
    # Should still work, just use the provided value
    assert result.exit_code == 0

    # Test zero cache max GB
    result = runner.invoke(
        app, ["ask", "total receipts for ANR", "--cache-max-gb", "0", "--dry-run"]
    )
    # Should still work, just use the provided value
    assert result.exit_code == 0


def test_all_materialize_strategies():
    """Test all materialize strategies work."""
    strategies = ["all", "heavy", "never"]

    for strategy in strategies:
        result = runner.invoke(
            app, ["ask", "total receipts for ANR", "--materialize", strategy, "--dry-run"]
        )
        assert result.exit_code == 0, f"Strategy {strategy} failed"
        assert "Plan Structure:" in result.stdout


def test_export_path_handling():
    """Test different export path handling scenarios."""
    # Test with custom path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        export_path = f.name

    try:
        result = runner.invoke(
            app, ["ask", "total receipts for ANR", "--export", export_path, "--dry-run"]
        )
        assert result.exit_code == 0
    finally:
        Path(export_path).unlink(missing_ok=True)

    # Test with auto path
    result = runner.invoke(app, ["ask", "total receipts for ANR", "--export", "auto", "--dry-run"])
    assert result.exit_code == 0


@pytest.mark.parametrize("command", ["plan", "run", "ask"])
def test_command_exists(command):
    """Test that all required commands exist and show help."""
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert command in result.stdout.lower() or "help" in result.stdout.lower()
