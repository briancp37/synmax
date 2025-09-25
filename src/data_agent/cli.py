"""CLI interface for the data agent using Typer."""

import time
from pathlib import Path
from typing import Literal, Optional

import orjson
import polars as pl
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
        from data_agent.ingest.loader import DEFAULT_DATA_PATH

        typer.echo(f"Loaded dataset from: {path or str(DEFAULT_DATA_PATH)}")
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
        "agent",
        "--planner",
        help="Planner type: agent (new DAG mode), deterministic, or llm (legacy)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="LLM model to use: gpt-4.1, gpt-5, claude-sonnet, claude-opus"
    ),
    export: Optional[str] = typer.Option(
        None,
        "--export",
        help="Export results to JSON file (use 'auto' for artifacts/outputs/{run_id}.json)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan JSON without executing"),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass cache and force fresh execution"
    ),
    fallback: bool = typer.Option(
        True,
        "--fallback/--no-fallback",
        help="Use deterministic fallbacks on LLM failure (agent mode only)",
    ),
) -> None:
    """Ask a natural-language question about the dataset."""
    import json

    # Ensure directories exist
    config.ensure_directories()

    # Load the dataset - use real data if available, otherwise golden dataset
    from data_agent.config import DATA_PATH
    from data_agent.ingest.loader import load_dataset

    if DATA_PATH.exists():
        lf = load_dataset(str(DATA_PATH), False)
    else:
        lf = load_dataset("examples/golden.parquet", False)

    try:
        if planner == "agent":
            # New agent planner mode (DAG-based)
            from data_agent.core.agent_planner import AgentPlanner, estimate_plan_complexity
            from data_agent.core.llm_client import LLMClient
            from data_agent.ingest.dictionary import load_data_dictionary

            typer.echo(f"Question: {q}")
            typer.echo("Using planner: agent (DAG mode)")
            if model:
                typer.echo(f"Using model: {model}")

            # Load data dictionary to get available columns
            data_dict = load_data_dictionary()
            available_columns = list(data_dict["schema"].keys())

            # Create LLM client if model specified
            client = None
            if model:
                client = LLMClient(model=model)  # type: ignore[arg-type]

            # Create planner and generate plan
            planner_instance = AgentPlanner(client)
            plan_graph = planner_instance.plan(q, available_columns, fallback=fallback)

            if dry_run:
                # Show detailed plan and estimates
                estimates = estimate_plan_complexity(plan_graph)
                typer.echo(f"\nPlan hash: {plan_graph.plan_hash()}")

                typer.echo("\nPlan Structure:")
                typer.echo(f"  Topological order: {' → '.join(estimates['topological_order'])}")
                typer.echo(f"  Estimated time: {estimates['estimated_time_seconds']:.1f}s")
                typer.echo(f"  Estimated memory: {estimates['estimated_memory_mb']}MB")
                if estimates["will_checkpoint"]:
                    typer.echo(f"  Will checkpoint: {', '.join(estimates['will_checkpoint'])}")

                typer.echo("\nPlan JSON:")
                typer.echo(json.dumps(plan_graph.model_dump(), indent=2))
                logger.info(
                    "Dry run executed (agent mode)",
                    extra={"question": q, "plan_hash": plan_graph.plan_hash()},
                )
                return

            # Execute the plan using agent_executor
            from data_agent.core.agent_executor import execute as agent_execute
            from data_agent.core.handles import StepHandle, StepStats

            # Create handle for the raw dataset
            dataset_path = config.DATA_PATH
            if not dataset_path.exists():
                typer.echo("Error: Dataset not found. Please run 'agent load' first.", err=True)
                raise typer.Exit(1) from None

            # Create dataset handle
            stats = StepStats(
                rows=lf.select(pl.len()).collect().item(),
                bytes=dataset_path.stat().st_size,
                columns=len(lf.columns),
                null_count={},
                computed_at=time.time(),
            )

            dataset_handle = StepHandle(
                id="raw",
                store="parquet",
                path=dataset_path,
                engine="polars",
                schema={col: str(dtype) for col, dtype in zip(lf.columns, lf.dtypes)},
                stats=stats,
                fingerprint="dataset",
            )

            # Execute the plan
            result_df, evidence = agent_execute(plan_graph, dataset_handle)

            typer.echo(f"\nPlan hash: {plan_graph.plan_hash()}")
            typer.echo(f"Steps executed: {len(evidence['steps'])}")
            typer.echo(f"Final result: {result_df.height} rows, {result_df.width} columns")

            # Show result table
            if result_df.height > 0:
                typer.echo("\nResult:")
                typer.echo(str(result_df))
            else:
                typer.echo("\nResult: Empty table")

            # Export if requested
            if export:
                export_path = export
                if export == "auto":
                    timestamp = int(time.time())
                    export_path = f"artifacts/outputs/{timestamp}_{plan_graph.plan_hash()[:8]}.json"

                export_data = {
                    "question": q,
                    "planner": "agent",
                    "model": model,
                    "plan": plan_graph.model_dump(),
                    "plan_hash": plan_graph.plan_hash(),
                    "result": {
                        "rows": result_df.height,
                        "columns": result_df.width,
                        "data": result_df.to_dicts() if result_df.height <= 100 else "truncated",
                    },
                    "evidence": evidence,
                }

                with open(export_path, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)
                typer.echo(f"\nPlan exported to: {export_path}")

            logger.info(
                "Ask command executed (agent mode)",
                extra={
                    "question": q,
                    "planner": planner,
                    "plan_hash": plan_graph.plan_hash(),
                },
            )
            return

        else:
            # Legacy planner modes (deterministic or llm)
            from data_agent.cache import CacheManager
            from data_agent.core.executor import run
            from data_agent.core.planner import plan as create_plan
            from data_agent.ingest.loader import load_dataset

            typer.echo(f"Question: {q}")
            typer.echo(f"Using planner: {planner}")
            if model:
                typer.echo(f"Using model: {model}")

            # Create the plan
            query_plan = create_plan(q, deterministic=(planner == "deterministic"), model=model)

            if dry_run:
                # Show plan JSON and exit
                plan_json = query_plan.model_dump()
                typer.echo("\nPlan JSON:")
                # Use custom serializer to handle Polars expressions
                typer.echo(json.dumps(plan_json, indent=2, default=str))
                logger.info("Dry run executed", extra={"question": q, "planner": planner})
                return

            # Dataset already loaded at the top of the function

            # Create cache manager (or None to bypass cache)
            cache_manager = None if no_cache else CacheManager()

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

            if answer.evidence["operation"]:
                op = answer.evidence["operation"]
                typer.echo(f"• Operation: {op['type']}")
                if op["parameters"]:
                    typer.echo("• Parameters:")
                    for key, value in op["parameters"].items():
                        typer.echo(f"  - {key}: {value}")
                    if op["type"] == "changepoint":
                        typer.echo(
                            "  (Tip: Ask 'with min_confidence=0.5' or 'with penalty=5.0' "
                            "to adjust parameters)",
                        )

            plan_time = answer.evidence["timings_ms"]["plan"]
            collect_time = answer.evidence["timings_ms"]["collect"]
            typer.echo(f"• Runtime: {plan_time:.1f}ms plan, {collect_time:.1f}ms collect")
            typer.echo(f"• Cache: {'hit' if answer.evidence['cache']['hit'] else 'miss'}")

            if export:
                # Export results to JSON using the export module
                from data_agent.core.export import export_results

                if export == "auto":
                    # Auto-generate filename in artifacts/outputs/
                    export_path = export_results(q, query_plan, answer)
                else:
                    # Use custom path
                    export_path = export_results(q, query_plan, answer, export_path=export)

                typer.echo(f"\nResults exported to: {export_path}")

            logger.info(
                "Ask command executed",
                extra={
                    "question": q,
                    "planner": planner,
                    "export": export,
                    "rows_out": answer.evidence["rows_out"],
                },
            )

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Make sure to run 'agent load' first to set up the dataset.")
        raise typer.Exit(1) from e
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
        raise typer.Exit(1) from None

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
    min_confidence: float = typer.Option(
        0.7, "--min-confidence", help="Minimum confidence threshold (0.0-2.0+, default 0.7)"
    ),
    export: Optional[str] = typer.Option(None, "--export", help="Export results to JSON file"),
) -> None:
    """Detect change-point events in the data."""
    try:
        from pathlib import Path

        from rich.console import Console
        from rich.table import Table

        from data_agent.core.events import build_event_card
        from data_agent.core.ops import apply_plan
        from data_agent.core.plan_schema import Filter, Plan
        from data_agent.ingest.loader import load_dataset

        console = Console()
        console.print("Detecting change-point events...")

        if pipeline:
            console.print(f"Pipeline: {pipeline}")
        if since:
            console.print(f"Since: {since}")
        console.print(f"Showing top {top} events with confidence >= {min_confidence}")
        if min_confidence == 0.7:
            console.print(
                "[dim](Use --min-confidence to adjust threshold, e.g., --min-confidence 0.5)[/dim]"
            )

        # Load dataset
        from data_agent.config import DATA_PATH

        dataset_path = None
        if DATA_PATH.exists():
            dataset_path = str(DATA_PATH)
        elif Path("examples/golden.parquet").exists():
            dataset_path = "examples/golden.parquet"
        else:
            console.print("[red]Error: No dataset found. Please run 'agent load' first.[/red]")
            raise typer.Exit(1) from None

        with console.status("Loading dataset..."):
            lf = load_dataset(path=dataset_path, auto=False)

        # Build filters based on options
        filters = []
        if pipeline:
            filters.append(Filter(column="pipeline_name", op="=", value=pipeline))
        if since:
            # Add date filter for since parameter
            filters.append(Filter(column="eff_gas_day", op="between", value=[since, "2030-01-01"]))

        # Create a plan for changepoint detection
        plan = Plan(
            filters=filters,
            op="changepoint",
            op_args={
                "groupby_cols": ["pipeline_name"] if not pipeline else None,
                "value_col": "scheduled_quantity",
                "date_col": "eff_gas_day",
                "min_size": 10,
                "penalty": 10.0,
                "min_confidence": min_confidence,
            },
        )

        with console.status("Analyzing data for change points..."):
            # Execute the plan
            result_lf = apply_plan(lf, plan)
            changepoints_df = result_lf.collect()

        if changepoints_df.is_empty():
            console.print(
                f"[yellow]No change points found with confidence >= {min_confidence}[/yellow]"
            )
            console.print(
                "[dim]Try lowering --min-confidence (e.g., --min-confidence 0.5) "
                "to see more results[/dim]"
            )
        else:
            # Sort by confidence and limit to top N
            sorted_df = changepoints_df.sort("confidence", descending=True).head(top)
            num_events = len(changepoints_df)
            console.print(
                f"[dim]Found {num_events} high-confidence change points (>= {min_confidence})[/dim]"
            )

            # Display results in a table
            table = Table(title="Change Point Events")
            table.add_column("Date", style="cyan")
            table.add_column("Before Mean", justify="right", style="green")
            table.add_column("After Mean", justify="right", style="red")
            table.add_column("Change %", justify="right", style="yellow")
            table.add_column("Confidence", justify="right", style="magenta")
            if "pipeline_name" in sorted_df.columns:
                table.add_column("Pipeline", style="blue")

            for row in sorted_df.iter_rows(named=True):
                change_pct = f"{row['change_magnitude']*100:.1f}%"
                confidence = f"{row['confidence']:.2f}"
                before_mean = f"{row['before_mean']:.1f}"
                after_mean = f"{row['after_mean']:.1f}"

                row_data = [
                    str(row["changepoint_date"]),
                    before_mean,
                    after_mean,
                    change_pct,
                    confidence,
                ]

                if "pipeline_name" in row:
                    row_data.append(row["pipeline_name"])

                table.add_row(*row_data)

            console.print(table)

            # Build event card for additional insights
            event_card = build_event_card(sorted_df, lf)
            console.print(f"\n[bold]Summary:[/bold] {event_card['summary']}")

            # Export if requested
            if export:
                export_data = {"changepoints": sorted_df.to_dicts(), "event_card": event_card}
                with open(export, "w") as f:
                    import json

                    json.dump(export_data, f, indent=2, default=str)
                console.print(f"Results exported to {export}")

        logger.info(
            "Events command executed",
            extra={
                "pipeline": pipeline,
                "since": since,
                "top": top,
                "min_confidence": min_confidence,
                "events_found": len(changepoints_df) if not changepoints_df.is_empty() else 0,
            },
        )

    except FileNotFoundError as e:
        console.print("[red]Error: Dataset not found. Please run 'agent load' first.[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error detecting events: {e}[/red]")
        logger.error("Events command failed", extra={"error": str(e)})
        raise typer.Exit(1) from e


@app.command()
def cluster(
    entity_type: str = typer.Option(
        ..., "--entity-type", help="Entity type to cluster: loc or counterparty"
    ),
    k: int = typer.Option(6, "--k", help="Number of clusters"),
    export: Optional[str] = typer.Option(None, "--export", help="Export results to JSON file"),
) -> None:
    """Cluster entities by behavior patterns."""
    valid_entity_types = ["loc", "counterparty"]
    if entity_type not in valid_entity_types:
        typer.echo(f"Invalid entity type. Choose from: {', '.join(valid_entity_types)}")
        raise typer.Exit(1) from None

    typer.echo(f"Clustering {entity_type} entities into {k} clusters...")

    try:
        # Load the dataset - use real data if available, otherwise golden dataset
        from data_agent.config import DATA_PATH
        from data_agent.core.cluster import cluster_entities
        from data_agent.ingest.loader import load_dataset

        if DATA_PATH.exists():
            lf = load_dataset(str(DATA_PATH), False)
        else:
            lf = load_dataset("examples/golden.parquet", False)

        # Run clustering
        # Type assertion to satisfy mypy
        entity_type_literal: Literal["loc", "counterparty"] = entity_type  # type: ignore[assignment]
        results_df, metrics = cluster_entities(lf, entity_type_literal, k, random_state=42)

        # Display results
        typer.echo("\nCluster Results:")
        typer.echo(
            results_df.group_by(["cluster_id", "cluster_name"])
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("mean_flow").mean().alias("avg_flow"),
                ]
            )
            .sort("cluster_id")
        )

        # Display metrics
        typer.echo("\nMetrics:")
        typer.echo(f"• Silhouette Score: {metrics['silhouette_score']:.3f}")
        typer.echo(f"• Entities Clustered: {metrics['n_entities']:,}")
        typer.echo(
            f"• Cluster Size Range: {metrics['min_cluster_size']}-{metrics['max_cluster_size']}"
        )

        # Export if requested
        if export:
            export_data = {
                "results": results_df.to_dicts(),
                "metrics": metrics,
                "parameters": {"entity_type": entity_type, "k": k},
            }
            Path(export).write_text(orjson.dumps(export_data, option=orjson.OPT_INDENT_2).decode())
            typer.echo(f"Results exported to {export}")

        logger.info(
            "Cluster command executed",
            extra={
                "entity_type": entity_type,
                "k": k,
                "silhouette_score": metrics["silhouette_score"],
                "n_entities": metrics["n_entities"],
            },
        )

    except Exception as e:
        typer.echo(f"Error during clustering: {e}")
        logger.error(
            "Clustering failed", extra={"error": str(e), "entity_type": entity_type, "k": k}
        )
        raise typer.Exit(1) from e


@app.command()
def plan(
    query: str = typer.Argument(..., help="Natural language question to plan"),
    model: Optional[str] = typer.Option(
        None, "--model", help="LLM model to use: gpt-4.1, gpt-5, claude-sonnet, claude-opus"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show plan and estimates without executing"
    ),
    export: Optional[str] = typer.Option(None, "--export", help="Export plan JSON to file"),
    fallback: bool = typer.Option(
        True, "--fallback/--no-fallback", help="Use deterministic fallbacks on LLM failure"
    ),
) -> None:
    """Plan a query using LLM and show the resulting DAG."""
    import json

    from data_agent.core.agent_planner import AgentPlanner, estimate_plan_complexity
    from data_agent.core.llm_client import LLMClient
    from data_agent.ingest.dictionary import load_data_dictionary

    try:
        # Load data dictionary to get available columns
        data_dict = load_data_dictionary()
        available_columns = list(data_dict["schema"].keys())

        # Create LLM client if model specified
        client = None
        if model:
            client = LLMClient(model=model)  # type: ignore[arg-type]

        # Create planner and generate plan
        planner = AgentPlanner(client)
        plan_graph = planner.plan(query, available_columns, fallback=fallback)

        # Show plan details
        typer.echo(f"Generated plan for: {query}")
        typer.echo(f"Plan hash: {plan_graph.plan_hash()}")
        typer.echo(f"Steps: {len(plan_graph.nodes)}")
        typer.echo(f"Edges: {len(plan_graph.edges)}")

        if dry_run:
            # Show detailed plan and estimates
            estimates = estimate_plan_complexity(plan_graph)
            typer.echo("\nPlan Structure:")
            typer.echo(f"  Topological order: {' → '.join(estimates['topological_order'])}")
            typer.echo(f"  Estimated time: {estimates['estimated_time_seconds']:.1f}s")
            typer.echo(f"  Estimated memory: {estimates['estimated_memory_mb']}MB")
            if estimates["will_checkpoint"]:
                typer.echo(f"  Will checkpoint: {', '.join(estimates['will_checkpoint'])}")

            typer.echo("\nSteps:")
            for step in plan_graph.nodes:
                typer.echo(f"  {step.id}: {step.op} {step.params}")

            typer.echo("\nEdges:")
            for edge in plan_graph.edges:
                typer.echo(f"  {edge.src} → {edge.dst}")

        # Export if requested
        if export:
            plan_data = plan_graph.model_dump()
            with open(export, "w") as f:
                json.dump(plan_data, f, indent=2)
            typer.echo(f"\nPlan exported to: {export}")

        logger.info(
            "Plan command executed",
            extra={
                "query": query,
                "model": model,
                "dry_run": dry_run,
                "plan_hash": plan_graph.plan_hash(),
            },
        )

    except FileNotFoundError:
        typer.echo("Error: Data dictionary not found. Run 'agent load' first.", err=True)
        raise typer.Exit(1) from None
    except Exception as e:
        typer.echo(f"Planning failed: {e}", err=True)
        logger.error("Plan command failed", extra={"error": str(e), "query": query})
        raise typer.Exit(1) from None


@app.command()
def run(
    plan_file: str = typer.Option(..., "--plan", help="Path to plan JSON file"),
    export: Optional[str] = typer.Option(
        None,
        "--export",
        help="Export results to JSON file (use 'auto' for artifacts/outputs/{hash}.json)",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass cache and force fresh execution"
    ),
) -> None:
    """Execute a pre-generated plan from JSON file."""
    import json
    from pathlib import Path

    from data_agent.core.agent_schema import PlanGraph
    from data_agent.ingest.loader import load_dataset

    try:
        # Load and validate plan
        plan_path = Path(plan_file)
        if not plan_path.exists():
            typer.echo(f"Error: Plan file not found: {plan_file}", err=True)
            raise typer.Exit(1) from None

        with open(plan_path) as f:
            plan_data = json.load(f)

        plan_graph = PlanGraph(**plan_data)

        # Execute the plan using agent_executor
        from data_agent.core.agent_executor import execute as agent_execute
        from data_agent.core.handles import StepHandle, StepStats
        from data_agent.ingest.loader import load_dataset

        # Load dataset
        lf = load_dataset(None, False)
        dataset_path = config.DATA_PATH
        if not dataset_path.exists():
            typer.echo("Error: Dataset not found. Please run 'agent load' first.", err=True)
            raise typer.Exit(1) from None

        # Create dataset handle
        stats = StepStats(
            rows=lf.select(pl.len()).collect().item(),
            bytes=dataset_path.stat().st_size,
            columns=len(lf.columns),
            null_count={},
            computed_at=__import__("time").time(),
        )

        dataset_handle = StepHandle(
            id="raw",
            store="parquet",
            path=dataset_path,
            engine="polars",
            schema={col: str(dtype) for col, dtype in zip(lf.columns, lf.dtypes)},
            stats=stats,
            fingerprint="dataset",
        )

        # Execute the plan
        result_df, evidence = agent_execute(plan_graph, dataset_handle)

        typer.echo(f"Plan executed: {plan_graph.plan_hash()}")
        typer.echo(f"Steps executed: {len(evidence['steps'])}")
        typer.echo(f"Final result: {result_df.height} rows, {result_df.width} columns")

        # Show result table
        if result_df.height > 0:
            typer.echo("\nResult:")
            typer.echo(str(result_df))
        else:
            typer.echo("\nResult: Empty table")

        # Export results if requested
        if export:
            output_data = {
                "plan": plan_graph.model_dump(),
                "plan_hash": plan_graph.plan_hash(),
                "result": {
                    "rows": result_df.height,
                    "columns": result_df.width,
                    "data": result_df.to_dicts() if result_df.height <= 100 else "truncated",
                },
                "evidence": evidence,
            }

            with open(export, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            typer.echo(f"\nResults exported to: {export}")

        logger.info(
            "Run command executed",
            extra={"plan_file": plan_file, "plan_hash": plan_graph.plan_hash()},
        )

    except Exception as e:
        typer.echo(f"Execution failed: {e}", err=True)
        logger.error("Run command failed", extra={"error": str(e), "plan_file": plan_file})
        raise typer.Exit(1) from None


@app.command()
def cache(
    clear: bool = typer.Option(False, "--clear", help="Clear the cache"),
    stats: bool = typer.Option(False, "--stats", help="Show cache statistics"),
    gc: bool = typer.Option(False, "--gc", help="Run garbage collection"),
) -> None:
    """Manage query result cache."""
    from data_agent.cache import CacheManager
    from data_agent.core.handles import HandleStorage

    # Only allow one operation at a time
    operations = sum([clear, stats, gc])
    if operations > 1:
        typer.echo("Cannot use multiple operations together")
        raise typer.Exit(1) from None

    cache_manager = CacheManager()
    handle_storage = HandleStorage()

    if clear:
        cache_manager.clear()
        # Also clear step handle storage
        if handle_storage.base_dir.exists():
            import shutil

            shutil.rmtree(handle_storage.base_dir)
            handle_storage.base_dir.mkdir(parents=True, exist_ok=True)
        typer.echo("Cache and step handles cleared")
        logger.info("Cache and step handles cleared")
    elif gc:
        # Run garbage collection on both cache and step handles
        cache_gc_stats = cache_manager.garbage_collect()
        handle_files_removed = handle_storage.cleanup_expired(config.CACHE_TTL_HOURS)

        typer.echo("Garbage Collection Results:")
        typer.echo(
            f"  Query cache: {cache_gc_stats['files_removed']} files removed, "
            f"{cache_gc_stats['bytes_freed_mb']:.1f}MB freed"
        )
        typer.echo(f"  Step handles: {handle_files_removed} files removed")

        # Show remaining stats
        cache_remaining = cache_manager.stats()
        handle_remaining = handle_storage.get_storage_stats()
        typer.echo(f"  Remaining cache: {cache_remaining['total_size_mb']:.1f}MB")
        typer.echo(f"  Remaining handles: {handle_remaining['total_size_mb']:.1f}MB")

        logger.info(
            "Garbage collection completed",
            extra={
                "cache_files_removed": cache_gc_stats,
                "handle_files_removed": handle_files_removed,
            },
        )
    elif stats:
        # Show stats for both cache and step handles
        cache_stats = cache_manager.stats()
        handle_stats = handle_storage.get_storage_stats()

        typer.echo("Cache Statistics:")
        typer.echo("  Query Results Cache:")
        typer.echo(f"    Total files: {cache_stats['total_files']}")
        typer.echo(f"    Parquet files: {cache_stats['parquet_files']}")
        typer.echo(f"    JSON files: {cache_stats['json_files']}")
        typer.echo(f"    Total size: {cache_stats['total_size_mb']:.1f}MB")
        typer.echo(f"    Valid entries: {cache_stats['valid_entries']}")
        typer.echo(f"    TTL: {cache_stats['ttl_hours']} hours")
        typer.echo(f"    Max size: {cache_stats['max_size_gb']}GB")

        typer.echo("  Step Handles Storage:")
        typer.echo(f"    Total files: {handle_stats['total_files']}")
        typer.echo(f"    Total size: {handle_stats['total_size_mb']:.1f}MB")
        typer.echo(f"    Directories: {handle_stats['directories']}")
        typer.echo(f"    Base dir: {handle_stats['base_dir']}")

        total_mb = cache_stats.get("total_size_mb", 0) + handle_stats.get("total_size_mb", 0)
        typer.echo(f"  Combined storage: {total_mb:.1f}MB")

        logger.info(
            "Cache stats requested",
            extra={"cache_stats": cache_stats, "handle_stats": handle_stats},
        )
    else:
        typer.echo(
            "Use --clear to clear cache, --stats to show statistics, "
            "or --gc to run garbage collection"
        )


if __name__ == "__main__":
    app()
