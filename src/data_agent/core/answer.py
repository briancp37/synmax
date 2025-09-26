"""LLM Answer Composer for generating concise written answers from query results."""

import json
from typing import Any

import polars as pl


def compose_answer_llm(
    plan: dict[str, Any], df: pl.DataFrame, evidence: dict, llm, style: str = "brief"
) -> str:
    """Generate a concise written answer from the final result table + plan + evidence.

    Args:
        plan: The query plan dictionary containing operation details
        df: The result DataFrame from query execution
        evidence: Evidence dictionary with filters and metadata
        llm: LLMClient instance for generating the answer
        style: Answer style, defaults to "brief"

    Returns:
        A concise 2-4 sentence answer based on the results
    """
    if df.height == 0:
        return "No results for the specified filters; see Evidence."

    # Get top 3 rows as structured data
    highlight_rows = df.head(min(3, df.height)).to_dicts()

    # Build compact prompt structure
    prompt_data = {
        "task": plan.get("op") or plan.get("macro") or "analysis",
        "filters": evidence.get("filters"),
        "method": plan.get("op"),
        "columns": df.columns,
        "highlights": highlight_rows,
        "style": style,
        "instructions": (
            "Write 2-4 sentences. Stay factual to highlights. "
            "Mention method if present. No extra claims."
        ),
    }

    # Create the prompt message
    prompt_text = f"""
Based on the following query results, write a concise answer:

Task: {prompt_data['task']}
Filters applied: {prompt_data['filters']}
Method used: {prompt_data['method']}
Result columns: {prompt_data['columns']}

Top results:
{json.dumps(prompt_data['highlights'], indent=2, default=str)}

Instructions: {prompt_data['instructions']}
Style: {prompt_data['style']}

Please provide a factual summary in 2-4 sentences that highlights the key findings from the data.
"""

    try:
        # Call the LLM to generate the answer
        response = llm.call(messages=[{"role": "user", "content": prompt_text}])
        return response.get("content", "Answer unavailable.")
    except Exception:
        return "Answer unavailable."
