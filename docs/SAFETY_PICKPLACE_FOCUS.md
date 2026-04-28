# Pickplace SafetyGuarantor Focus Suite

This suite evaluates approach-zone safety stability on pickplace-only scenarios
with an OpenAI-compatible DeepSeek backend. It is intended for external
execution only; do not run it from Codex when the task is limited to generating
configuration and documentation.

## Suite

`configs/experiment_suites/safety_guarantor_pickplace_focus_deepseek.yaml`
selects 20 scenarios from `configs/generated/pickplace_hard/manifest.json`.

The suite runs three methods:

- `llm_only_multi_bt`
- `llm_rule_with_repair_no_approach`
- `llm_rule_with_repair_full`

Total size: 20 scenarios x 3 methods = 60 runs.

## Environment

Set the API key in your shell or secret manager. Do not commit it and do not
paste it into logs.

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-v4-flash"
export OPENAI_PROVIDER_PROFILE="deepseek"
```

## Full Run

```bash
python -m src.experiments.run_safety_ablation_experiment \
  --suite configs/experiment_suites/safety_guarantor_pickplace_focus_deepseek.yaml \
  --backend openai-compatible \
  --output-dir results/safety_guarantor_pickplace_focus_deepseek
```

## Outputs

The runner writes:

- `raw.jsonl`
- `summary.csv`
- `summary.md`
- `traces/*.json`

The focus summary metrics are:

- `success_rate`
- `avg_num_iters`
- `xml_valid_rate`
- `resource_request_valid_rate`
- `timeout_rate`
- `makespan`
- `collision_event_count`
- `actual_collision_count`
- `resource_conflict_count`
- `approach_zone_wait_count`
- `rule_prevented_approach_conflict_count`
- `lock_wait_count`
- `provider_error_rate`

The runner may also emit additional existing summary fields. For this suite,
the main comparison is between repair without approach-zone locking and the
full method with approach-zone locking enabled.

## Validation

Validate code importability without calling a real LLM API:

```bash
python -m compileall src tests
```
