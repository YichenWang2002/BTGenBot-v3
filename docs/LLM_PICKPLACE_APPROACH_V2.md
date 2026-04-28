# LLM Pickplace Approach V2

This suite evaluates whether fine-grained approach-zone locking helps on real
LLM-generated pick/place behavior trees.

## Suite

The suite is defined in:

```bash
configs/experiment_suites/llm_pickplace_approach_v2.yaml
```

It uses the first 10 pickplace scenarios from
`configs/generated/pickplace_hard/manifest.json` and runs two methods per
scenario:

- `centralized_rule_multi_bt`
- `centralized_rule_with_approach_radius_1`

`centralized_rule_multi_bt` uses the original centralized rule behavior.
`centralized_rule_with_approach_radius_1` enables cell, edge, resource,
recovery, and approach-zone locks with approach radius 1.
The exact coordination settings are recorded under `method_configs` in the
suite YAML.

## API-Free Validation

Use the mock backend to validate the suite without calling any LLM API:

```bash
python -m src.repair.refinement_loop \
  --suite configs/experiment_suites/llm_pickplace_approach_v2.yaml \
  --backend mock \
  --max-iters 3 \
  --output-dir results/llm_pickplace_approach_v2_mock
```

## Real Backend Run

Only run this when you explicitly want to call an OpenAI-compatible provider:

```bash
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."

python -m src.repair.refinement_loop \
  --suite configs/experiment_suites/llm_pickplace_approach_v2.yaml \
  --backend openai-compatible \
  --max-iters 3 \
  --output-dir results/llm_pickplace_approach_v2_openai_compatible
```

## Outputs

Each run writes:

- `raw.jsonl`
- `summary.csv`
- `summary.md`
- `runs/<run_id>/iter_<n>/prompt.txt`
- `runs/<run_id>/iter_<n>/raw_response.txt`
- `runs/<run_id>/iter_<n>/parsed_response.json`
- `runs/<run_id>/iter_<n>/static_validation.json`
- `runs/<run_id>/iter_<n>/failure_report.json`
- `runs/<run_id>/iter_<n>/metrics.json`
- `runs/<run_id>/iter_<n>/trace.json`

The summary includes success rate, average iterations, collision/resource
conflict counts, approach-zone wait and lock metrics, timeout rate, and provider
error rate.

For paper-facing collision claims, use the final-trace audit from
`src.analysis.audit_trace_collisions` and the metric definitions in
`docs/METRICS.md`. The legacy `collision_count` field is retained for
compatibility; report `collision_event_count` and `actual_collision_count` when
distinguishing simulator-reported events from trace-audited trajectory
collisions.

This suite does not modify `dataset/bt_dataset.json`.
