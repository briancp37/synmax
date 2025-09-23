"""CLI interface for the data agent using Typer."""

from typing import Optional

import orjson
import typer

from data_agent import config
from data_agent.utils.logging import get_logger, setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SynMax Data Agent - Natural language queries over gas pipeline data",
)

# Set up logging on module import
setup_logging()
logger = get_logger("cli")


@app.command()
def load(
    path: Optional[str] = typer.Option(None, "--path", help="Path to parquet file to load"),
    auto: bool = typer.Option(
        False, "--auto", help="Auto-download dataset to ./data/ if path not provided"
    ),
) -> None:
    """Load dataset from --path or auto-download to ./data."""
    from data_agent.ingest.dictionary import build_data_dictionary, write_dictionary
    from data_agent.ingest.loader import load_dataset
    from data_agent.ingest.rollups import build_daily_rollups, write_daily_rollups

    # Ensure directories exist
    config.ensure_directories()

    try:
        # Load the dataset
        lf = load_dataset(path, auto)

        # Build and write data dictionary
        data_dict = build_data_dictionary(lf)
        write_dictionary(data_dict)

        # Build and write daily rollups
        rollups_df = build_daily_rollups(lf)
        rollups_path = write_daily_rollups(rollups_df)

        # Print schema information
        typer.echo(f"Loaded dataset from: {path or 'data/data.parquet'}")
        typer.echo(f"Rows: {data_dict['n_rows']:,}")
        typer.echo(f"Columns: {len(data_dict['schema'])}")
        typer.echo("\nSchema:")
        for col, dtype in data_dict["schema"].items():
            null_rate = data_dict["null_rates"][col]
            typer.echo(f"  {col}: {dtype} (null rate: {null_rate:.1%})")

        typer.echo("\nData dictionary written to: artifacts/data_dictionary.json")
        typer.echo(f"Daily rollups written to: {rollups_path}")

        logger.info(
            "Dataset load command executed",
            extra={"path": path, "auto": auto, "rows": data_dict["n_rows"]},
        )

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        logger.error("Load command failed", extra={"error": str(e), "path": path, "auto": auto})
        raise typer.Exit(1) from e


@app.command()
def ask(
    q: str = typer.Argument(..., help="Natural language question about the dataset"),
    planner: str = typer.Option(
        "deterministic", "--planner", help="Planner type: deterministic or llm"
    ),
    export: Optional[str] = typer.Option(None, "--export", help="Export results to JSON file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan JSON without executing"),
) -> None:
    """Ask a natural-language question about the dataset."""
    import json

    from data_agent.cache import CacheManager
    from data_agent.core.executor import run
    from data_agent.core.planner import plan as create_plan
    from data_agent.ingest.loader import load_dataset

    typer.echo(f"Question: {q}")
    typer.echo(f"Using planner: {planner}")

    try:
        # Create the plan
        query_plan = create_plan(q, deterministic=(planner == "deterministic"))

        if dry_run:
            # Show plan JSON and exit
            plan_json = query_plan.model_dump()
            typer.echo("\nPlan JSON:")
            # Use custom serializer to handle Polars expressions
            typer.echo(json.dumps(plan_json, indent=2, default=str))
            logger.info("Dry run executed", extra={"question": q, "planner": planner})
            return

        # Load the dataset - use real data if available, otherwise golden dataset
        from data_agent.config import DATA_PATH

        if DATA_PATH.exists():
            lf = load_dataset(str(DATA_PATH), False)
        else:
            lf = load_dataset("examples/golden.parquet", False)

        # Create cache manager
        cache_manager = CacheManager()

        # Execute the plan
        answer = run(lf, query_plan, cache_manager)

        # Display results
        typer.echo("\nAnswer:")
        typer.echo(answer.table)

        # Display evidence card
        typer.echo("\nEvidence Card:")
        typer.echo(f"• Rows out: {answer.evidence['rows_out']:,}")
        typer.echo(f"• Columns: {', '.join(answer.evidence['columns'])}")

        if answer.evidence["filters"]:
            typer.echo("• Filters applied:")
            for f in answer.evidence["filters"]:
                typer.echo(f"  - {f['column']} {f['op']} {f['value']}")

        if answer.evidence["aggregate"]:
            agg = answer.evidence["aggregate"]
            if agg["groupby"]:
                typer.echo(f"• Grouped by: {', '.join(agg['groupby'])}")
            typer.echo("• Metrics:")
            for m in agg["metrics"]:
                typer.echo(f"  - {m['fn']}({m['col']})")

        if answer.evidence["sort"]:
            sort = answer.evidence["sort"]
            typer.echo(
                f"• Sorted by: {', '.join(sort['by'])} ({'desc' if sort['desc'] else 'asc'})"
            )
            if sort["limit"]:
                typer.echo(f"• Limited to: {sort['limit']} rows")

        plan_time = answer.evidence["timings_ms"]["plan"]
        collect_time = answer.evidence["timings_ms"]["collect"]
        typer.echo(f"• Runtime: {plan_time:.1f}ms plan, {collect_time:.1f}ms collect")
        typer.echo(f"• Cache: {'hit' if answer.evidence['cache']['hit'] else 'miss'}")

        if export:
            # Export results to JSON
            # Convert plan to dict and serialize with orjson to handle Polars expressions
            plan_dict = query_plan.model_dump()
            export_data = {
                "question": q,
                "plan": orjson.loads(orjson.dumps(plan_dict, default=str)),
                "answer": {"table": answer.table.to_dicts(), "evidence": answer.evidence},
            }
            with open(export, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            typer.echo(f"\nResults exported to: {export}")

        logger.info(
            "Ask command executed",
            extra={
                "question": q,
                "planner": planner,
                "export": export,
                "rows_out": answer.evidence["rows_out"],
            },
        )

    except NotImplementedError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.error(
            "Ask command failed", extra={"error": str(e), "question": q, "planner": planner}
        )
        raise typer.Exit(1) from e


@app.command()
def rules(
    pipeline: Optional[str] = typer.Option(
        None, "--pipeline", help="Filter rules by pipeline name"
    ),
    since: Optional[str] = typer.Option(None, "--since", help="Show rules since date (YYYY-MM-DD)"),
) -> None:
    """Run data quality rules and show violations."""
    try:
        from pathlib import Path

        from rich.console import Console
        from rich.table import Table

        from data_agent.ingest.loader import load_dataset
        from data_agent.rules.engine import run_rules

        console = Console()

        # Try to load from default path first, then fallback to examples/golden.parquet
        from data_agent.config import DATA_PATH

        dataset_path = None
        if DATA_PATH.exists():
            dataset_path = str(DATA_PATH)
        elif Path("examples/golden.parquet").exists():
            dataset_path = "examples/golden.parquet"

        with console.status("Loading dataset..."):
            lf = load_dataset(path=dataset_path, auto=False)

        console.print("Running data quality rules...")
        if pipeline:
            console.print(f"Filtering by pipeline: {pipeline}")
        if since:
            console.print(f"Since date: {since}")

        with console.status("Analyzing data..."):
            results = run_rules(lf, pipeline=pipeline, since=since)

        # Create a table to display results
        table = Table(title="Data Quality Rules Summary")
        table.add_column("Rule ID", style="cyan", no_wrap=True)
        table.add_column("Description", style="magenta")
        table.add_column("Violations", justify="right", style="red")
        table.add_column("Sample Count", justify="right", style="yellow")

        # Rule descriptions
        rule_descriptions = {
            "R-001": "Missing Geo on Active",
            "R-002": "Duplicate loc_name across states",
            "R-003": "Zero-quantity streaks",
            "R-004": "Pipeline Imbalance (daily)",
            "R-005": "Schema mismatch",
            "R-006": "Eff gas day gaps",
        }

        total_violations = 0
        for rule_id, result in results.items():
            count = result["count"]
            samples_count = len(result["samples"])
            total_violations += count

            table.add_row(
                rule_id,
                rule_descriptions.get(rule_id, "Unknown rule"),
                str(count),
                str(samples_count),
            )

        console.print(table)
        console.print(f"\n[bold]Total violations found: {total_violations}[/bold]")

        # Show samples for rules with violations
        for rule_id, result in results.items():
            if result["count"] > 0 and result["samples"]:
                rule_desc = rule_descriptions.get(rule_id, "Unknown rule")
                console.print(f"\n[bold cyan]{rule_id} - {rule_desc} Samples:[/bold cyan]")
                for i, sample in enumerate(result["samples"][:3], 1):
                    console.print(f"  Sample {i}: {sample}")

        logger.info(
            "Rules command executed",
            extra={"pipeline": pipeline, "since": since, "total_violations": total_violations},
        )

    except FileNotFoundError as e:
        console.print("[red]Error: Dataset not found. Please run 'agent load' first.[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error running rules: {e}[/red]")
        logger.error("Rules command failed", extra={"error": str(e)})
        raise typer.Exit(1) from e


@app.command()
def metrics(
    name: str = typer.Option(..., "--name", help="Metric name: ramp_risk, reversal, or imbalance"),
    filters: Optional[str] = typer.Option(None, "--filters", help="Additional filters as JSON"),
) -> None:
    """Compute analytics metrics."""
    valid_metrics = ["ramp_risk", "reversal", "imbalance"]
    if name not in valid_metrics:
        typer.echo(f"Invalid metric. Choose from: {', '.join(valid_metrics)}")
        raise typer.Exit(1)

    typer.echo(f"Computing metric: {name}")
    if filters:
        typer.echo(f"With filters: {filters}")

    # TODO: analytics.compute_metric(name, filters)
    logger.info("Metrics command executed", extra={"metric_name": name, "filters": filters})
    typer.echo(f"Metric {name} (placeholder)")


@app.command()
def events(
    pipeline: Optional[str] = typer.Option(
        None, "--pipeline", help="Pipeline name to analyze for events"
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="Find events since date (YYYY-MM-DD)"
    ),
    top: int = typer.Option(10, "--top", help="Number of top events to show"),
) -> None:
    """Detect change-point events in the data."""
    typer.echo("Detecting change-point events...")

    if pipeline:
        typer.echo(f"Pipeline: {pipeline}")
    if since:
        typer.echo(f"Since: {since}")
    typer.echo(f"Showing top {top} events")

    # TODO: changepoint.detect_events(pipeline, since, top)
    logger.info("Events command executed", extra={"pipeline": pipeline, "since": since, "top": top})
    typer.echo("Change-point events (placeholder)")


@app.command()
def cluster(
    entity_type: str = typer.Option(
        ..., "--entity-type", help="Entity type to cluster: loc or counterparty"
    ),
    k: int = typer.Option(6, "--k", help="Number of clusters"),
) -> None:
    """Cluster entities by behavior patterns."""
    valid_entity_types = ["loc", "counterparty"]
    if entity_type not in valid_entity_types:
        typer.echo(f"Invalid entity type. Choose from: {', '.join(valid_entity_types)}")
        raise typer.Exit(1)

    typer.echo(f"Clustering {entity_type} entities into {k} clusters...")

    # TODO: clustering.cluster_entities(entity_type, k)
    logger.info("Cluster command executed", extra={"entity_type": entity_type, "k": k})
    typer.echo(f"Clustered {entity_type} entities (placeholder)")


@app.command()
def cache(
    clear: bool = typer.Option(False, "--clear", help="Clear the cache"),
    stats: bool = typer.Option(False, "--stats", help="Show cache statistics"),
) -> None:
    """Manage query result cache."""
    from data_agent.cache import CacheManager

    if clear and stats:
        typer.echo("Cannot use --clear and --stats together")
        raise typer.Exit(1)

    cache_manager = CacheManager()

    if clear:
        cache_manager.clear()
        typer.echo("Cache cleared")
        logger.info("Cache cleared")
    elif stats:
        stats_data = cache_manager.stats()
        typer.echo("Cache Statistics:")
        typer.echo(f"  Total files: {stats_data['total_files']}")
        typer.echo(f"  Parquet files: {stats_data['parquet_files']}")
        typer.echo(f"  JSON files: {stats_data['json_files']}")
        typer.echo(f"  Total size: {stats_data['total_size_bytes']:,} bytes")
        typer.echo(f"  Valid entries: {stats_data['valid_entries']}")
        typer.echo(f"  TTL: {stats_data['ttl_hours']} hours")
        logger.info("Cache stats requested", extra=stats_data)
    else:
        typer.echo("Use --clear to clear cache or --stats to show statistics")


if __name__ == "__main__":
    app()
