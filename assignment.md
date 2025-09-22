# Data Agent Project

## Overview
Create a **chat-based agent in Python** that can interact with the linked dataset and answer user questions. No GUI or web API required—CLI is fine. Use all the coding tools, LLMs and resources you want, there are no restrictions on what you are allowed to use to complete this project provided you can bring those same tools with you to SynMax should you be hired. The agent should handle questions from simple retrieval to advanced analysis, including:
Pattern recognition (e.g., clustering, correlations, trends)


Anomaly detection (e.g., outliers, rule violations)


Proposing plausible causes behind observed relationships (with caveats and evidence)


Dataset: https://drive.google.com/file/d/109vhmnSLN3oofjFdyb58l_rUZRa0d6C8/view?usp=drivesdk  Due: one week from receipt
Do not include the dataset file in your repo.
What to Build
Capabilities
Your agent should:
Ingest the dataset (infers schema & types; handles missing values).


Understand natural-language questions.


Plan & execute analysis over the data.


Return a concise answer plus supporting evidence (methods used, selected columns, filters).


Handle both deterministic queries (e.g., “count of … in 2024”) and analytic tasks (patterns, anomalies, causal hypotheses with limitations).
Engineering Requirements
Language: Python 3.10+


No front end / API required.


LLM options OpenAI & Anthropic


Do not upload the dataset to GitHub. Your code should either:


prompt for a local path, or


download from the link at runtime and store under ./data/ (ensure it’s .gitignore’d).


Setup & Keys
I will provide my own keys via environment variables.
Environment variables (use whichever you support):


OPENAI_API_KEY


ANTHROPIC_API_KEY


Include README sections:


Installation & quick start


How to supply the dataset path or enable auto-download


Example queries/outputs (anything your impressed with and want me to see)


Any assumptions or limitations


Dockerization is not required.
Deliverables
Public GitHub repo link (code only, no dataset).


README.md with clear installation & usage.


requirements.txt or pyproject.toml.


Evaluation (I provide the queries)
I will run a set of pre-written queries. Each query is scored on:
Accuracy (70%)


Correctness of numbers/tables/claims


Sound methodology for patterns & anomalies


Reasonable, evidence-backed causal hypotheses (not assertions)


Speed (30%)


Lower latency is better (measured per query)


Bonus: extra credit for particularly insightful or actionable findings, e.g.,
surfacing non-obvious segments/clusters with business interpretation,


detecting data quality issues that change conclusions,


identifying potential confounders or validating with simple robustness checks.
