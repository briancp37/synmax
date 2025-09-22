"""CLI interface for the data agent using Typer."""

import typer
from typing import Optional
from data_agent import config
from data_agent.utils.logging import setup_logging, get_logger

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SynMax Data Agent - Natural language queries over gas pipeline data"
)

# Set up logging on module import
setup_logging()
logger = get_logger("cli")

@app.command()
def load(
    path: Optional[str] = typer.Option(
        None, 
        "--path", 
        help="Path to parquet file to load"
    ),
    auto: bool = typer.Option(
        False, 
        "--auto", 
        help="Auto-download dataset to ./data/ if path not provided"
    )
) -> None:
    """Load dataset from --path or auto-download to ./data."""
    # Ensure directories exist
    config.ensure_directories()
    
    if path:
        typer.echo(f"Loading dataset from: {path}")
    elif auto:
        typer.echo("Auto-downloading dataset to ./data/")
        # TODO: call ingest.loader.load_dataset(path=None, auto=True)
    else:
        typer.echo("Please provide --path to a parquet file or use --auto to download")
        raise typer.Exit(1)
    
    logger.info("Dataset load command executed", extra={"path": path, "auto": auto})
    typer.echo("Loaded dataset (placeholder).")

@app.command()
def ask(
    q: str = typer.Argument(..., help="Natural language question about the dataset"),
    planner: str = typer.Option(
        "deterministic", 
        "--planner",
        help="Planner type: deterministic or llm"
    ),
    export: Optional[str] = typer.Option(
        None,
        "--export", 
        help="Export results to JSON file"
    )
) -> None:
    """Ask a natural-language question about the dataset."""
    typer.echo(f"Question: {q}")
    typer.echo(f"Using planner: {planner}")
    
    if export:
        typer.echo(f"Will export results to: {export}")
    
    # TODO: planner.plan(q) -> Plan; executor.run(Plan) -> Answer+Evidence
    logger.info(
        "Ask command executed", 
        extra={"question": q, "planner": planner, "export": export}
    )
    typer.echo("Answer (placeholder)")

@app.command()
def rules(
    pipeline: Optional[str] = typer.Option(
        None,
        "--pipeline",
        help="Filter rules by pipeline name"
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Show rules since date (YYYY-MM-DD)"
    )
) -> None:
    """Run data quality rules and show violations."""
    typer.echo("Running data quality rules...")
    
    if pipeline:
        typer.echo(f"Filtering by pipeline: {pipeline}")
    if since:
        typer.echo(f"Since date: {since}")
    
    # TODO: rules.engine.run_rules(pipeline=pipeline, since=since)
    logger.info(
        "Rules command executed",
        extra={"pipeline": pipeline, "since": since}
    )
    typer.echo("Rules scan (placeholder)")

@app.command()
def metrics(
    name: str = typer.Option(
        ...,
        "--name",
        help="Metric name: ramp_risk, reversal, or imbalance"
    ),
    filters: Optional[str] = typer.Option(
        None,
        "--filters",
        help="Additional filters as JSON"
    )
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
    logger.info(
        "Metrics command executed",
        extra={"metric_name": name, "filters": filters}
    )
    typer.echo(f"Metric {name} (placeholder)")

@app.command()
def events(
    pipeline: Optional[str] = typer.Option(
        None,
        "--pipeline",
        help="Pipeline name to analyze for events"
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Find events since date (YYYY-MM-DD)"
    ),
    top: int = typer.Option(
        10,
        "--top",
        help="Number of top events to show"
    )
) -> None:
    """Detect change-point events in the data."""
    typer.echo("Detecting change-point events...")
    
    if pipeline:
        typer.echo(f"Pipeline: {pipeline}")
    if since:
        typer.echo(f"Since: {since}")
    typer.echo(f"Showing top {top} events")
    
    # TODO: changepoint.detect_events(pipeline, since, top)
    logger.info(
        "Events command executed",
        extra={"pipeline": pipeline, "since": since, "top": top}
    )
    typer.echo("Change-point events (placeholder)")

@app.command()
def cluster(
    entity_type: str = typer.Option(
        ...,
        "--entity-type",
        help="Entity type to cluster: loc or counterparty"
    ),
    k: int = typer.Option(
        6,
        "--k",
        help="Number of clusters"
    )
) -> None:
    """Cluster entities by behavior patterns."""
    valid_entity_types = ["loc", "counterparty"]
    if entity_type not in valid_entity_types:
        typer.echo(f"Invalid entity type. Choose from: {', '.join(valid_entity_types)}")
        raise typer.Exit(1)
    
    typer.echo(f"Clustering {entity_type} entities into {k} clusters...")
    
    # TODO: clustering.cluster_entities(entity_type, k)
    logger.info(
        "Cluster command executed",
        extra={"entity_type": entity_type, "k": k}
    )
    typer.echo(f"Clustered {entity_type} entities (placeholder)")

@app.command()
def cache(
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear the cache"
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        help="Show cache statistics"
    )
) -> None:
    """Manage query result cache."""
    if clear and stats:
        typer.echo("Cannot use --clear and --stats together")
        raise typer.Exit(1)
    
    if clear:
        # TODO: cache.clear_cache()
        typer.echo("Cache cleared")
        logger.info("Cache cleared")
    elif stats:
        # TODO: cache.get_stats()
        typer.echo("Cache statistics (placeholder)")
        logger.info("Cache stats requested")
    else:
        typer.echo("Use --clear to clear cache or --stats to show statistics")

if __name__ == "__main__":
    app()
