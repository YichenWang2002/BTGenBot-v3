# SafetyGuarantor DeepSeek Ablation

This suite runs the SafetyGuarantor ablation with an OpenAI-compatible DeepSeek
backend. It is intended for external execution only; do not run it from Codex
when the task is limited to configuration and documentation.

## Suite

`configs/experiment_suites/safety_guarantor_ablation_deepseek_small.yaml`
selects:

- 5 scenarios from `nav_easy`
- 5 scenarios from `recovery_medium`
- 5 scenarios from `pickplace_hard`

The suite runs all five methods:

- `llm_only_multi_bt`
- `llm_static_critic_only`
- `llm_rule_no_repair`
- `llm_rule_with_repair_no_approach`
- `llm_rule_with_repair_full`

Total size: 15 scenarios x 5 methods = 75 runs.

## Environment

Set the API key in your shell or secret manager. Do not commit it and do not
paste it into logs.

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-v4-flash"
export OPENAI_PROVIDER_PROFILE="deepseek"
```

## Check Run

Run a 5-run check before launching the full suite:

```bash
python -m src.experiments.run_safety_ablation_experiment \
  --suite configs/experiment_suites/safety_guarantor_ablation_deepseek_small.yaml \
  --backend openai-compatible \
  --max-runs 5 \
  --output-dir results/safety_guarantor_ablation_deepseek_check5
```

`--max-runs 5` only proves that the API path works. It does not cover all
scenario families because it truncates the global run list.

## Balanced Check

Use a balanced check to sample each `scenario_family` before the full run:

```bash
python -m src.experiments.run_safety_ablation_experiment \
  --suite configs/experiment_suites/safety_guarantor_ablation_deepseek_small.yaml \
  --backend openai-compatible \
  --balanced-check 1 \
  --output-dir results/safety_guarantor_ablation_deepseek_check15
```

With `--balanced-check 1`, the runner selects one scenario from each family and
runs all five methods:

- `nav`: 1 scenario x 5 methods
- `recovery`: 1 scenario x 5 methods
- `pickplace`: 1 scenario x 5 methods

Total: 15 runs.

## Full Run

```bash
python -m src.experiments.run_safety_ablation_experiment \
  --suite configs/experiment_suites/safety_guarantor_ablation_deepseek_small.yaml \
  --backend openai-compatible \
  --output-dir results/safety_guarantor_ablation_deepseek_small
```

## Outputs

The runner writes:

- `raw.jsonl`
- `summary.csv`
- `summary.md`
- `traces/*.json`

The summary includes `overall` and `by_family` rows. Rows include the backend,
model name, `provider_error_rate`, and the SafetyGuarantor metrics used by the
mock ablation suite.

If the provider request fails, the run is recorded with `provider_error=true`,
a sanitized `error_message`, and an empty trace artifact. The runner must not
print or persist the API key.
