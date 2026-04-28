# LLM Evaluation

This project supports an API-free mock backend and an OpenAI-compatible backend for
LLM-generated multi-robot BT repair experiments.

## Build the Small Suite

Generate the 60-run small benchmark suite from the generated scenario manifests:

```bash
python -m src.repair.build_llm_suite \
  --nav-manifest configs/generated/nav_easy/manifest.json \
  --recovery-manifest configs/generated/recovery_medium/manifest.json \
  --pickplace-manifest configs/generated/pickplace_hard/manifest.json \
  --out configs/experiment_suites/llm_repair_small.yaml \
  --per-family 10 \
  --output-dir results/llm_repair_small_openai_compatible
```

The builder selects up to 10 existing scenario paths from each family in manifest
order and writes:

- `configs/experiment_suites/llm_repair_small.yaml`
- `configs/experiment_suites/llm_repair_small_manifest_summary.json`

## Run With Mock Backend

Use mock when no API key is available:

```bash
python -m src.repair.refinement_loop \
  --suite configs/experiment_suites/llm_repair_small.yaml \
  --backend mock \
  --max-iters 3 \
  --output-dir results/llm_repair_small_mock
```

## Run With DeepSeek / OpenAI-Compatible Backend

Set the OpenAI-compatible environment variables:

```bash
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="deepseek-v4-pro"
```

Then run:

```bash
python -m src.repair.refinement_loop \
  --suite configs/experiment_suites/llm_repair_small.yaml \
  --backend openai-compatible \
  --max-iters 3 \
  --output-dir results/llm_repair_small_openai_compatible
```

If any required environment variable is missing, the runner prints the export
commands and exits without calling the API.

## Outputs

Each benchmark output directory contains:

- `raw.jsonl`: one row per method/scenario run.
- `summary.csv`: aggregate rows by `method/backend/scenario_family` and overall
  rows by `method/backend`.
- `summary.md`: the same aggregates in Markdown.
- `runs/`: per-iteration prompts, raw responses, parsed JSON, static validation,
  metrics, traces, and failure reports.

## Metric Interpretation

- `resource_conflict_count`: actual resource conflicts, mainly in naive or
  LLM-only runs where multiple robots attempt to use the same object or zone.
- `resource_request_denied_count`: resource requests denied by the centralized
  resource manager.
- `lock_wait_count`: times a robot waited because a centralized resource lock was
  already held.
- `rule_prevented_resource_conflict_count`: potential resource conflicts blocked
  by centralized rules before becoming actual conflicts.

The smoke results show that `centralized_rule_multi_bt` can replace actual
resource conflicts with lock waits. `llm_only_multi_robot` may still show
recovery or resource conflicts because it does not use centralized arbitration.
