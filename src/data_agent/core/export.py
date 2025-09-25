"""Export functionality for data agent results."""

import json
import uuid
from datetime import datetime

import orjson

from data_agent.config import get_artifacts_dir
from data_agent.core.executor import Answer
from data_agent.core.plan_schema import Plan


def generate_run_id() -> str:
    """Generate a unique run ID for the export."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


def export_results(
    question: str,
    plan: Plan,
    answer: Answer,
    run_id: str | None = None,
    export_path: str | None = None,
) -> str:
    """Export query results to JSON file.

    Args:
        question: The natural language question
        plan: The execution plan
        answer: The answer with results and evidence
        run_id: Optional run ID, will be generated if not provided
        export_path: Optional custom export path, defaults to artifacts/outputs/{run_id}.json

    Returns:
        Path to the exported file
    """
    if run_id is None:
        run_id = generate_run_id()

    if export_path is None:
        # Ensure artifacts/outputs directory exists
        artifacts_dir = get_artifacts_dir()
        outputs_dir = artifacts_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        export_path = str(outputs_dir / f"{run_id}.json")

    # Convert plan to dict and serialize with orjson to handle Polars expressions
    plan_dict = plan.model_dump()

    # Prepare export data
    export_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "plan": orjson.loads(orjson.dumps(plan_dict, default=str)),
        "answer": {"table": answer.table.to_dicts(), "evidence": answer.evidence},
        "metadata": {
            "rows_returned": len(answer.table),
            "columns": answer.table.columns,
            "plan_type": getattr(plan, "op", None) or "basic",
            "cache_hit": answer.evidence.get("cache", {}).get("hit", False),
            "execution_time_ms": {
                "plan": answer.evidence.get("timings_ms", {}).get("plan", 0),
                "collect": answer.evidence.get("timings_ms", {}).get("collect", 0),
                "total": answer.evidence.get("timings_ms", {}).get("plan", 0)
                + answer.evidence.get("timings_ms", {}).get("collect", 0),
            },
        },
    }

    # Write to file
    with open(export_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return export_path
