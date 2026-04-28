# SafetyGuarantor Ablation

This suite validates the method hierarchy used in the paper without calling a
real LLM API. It uses deterministic mock BT generation on existing non-DAG
navigation, recovery, and pick/place scenarios.

## Methods

`llm_only_multi_bt` runs the mock LLM output directly. It does not hard-gate on
the static critic and does not enable centralized rules.

`llm_static_critic_only` enables the static critic and fails the run when static
validation fails. It still executes without centralized runtime safety.

`llm_rule_no_repair` enables the static critic plus SafetyGuarantor motion and
semantic resource layers. It disables repair and approach-zone locking.

`llm_rule_with_repair_no_approach` enables the static critic, SafetyGuarantor
motion and semantic resource layers, and a failure-driven repair loop. It does
not enable approach-zone locking.

`llm_rule_with_repair_full` is the full method. It enables repair plus motion,
semantic resource, and manipulation-area safety with `approach_radius=1`,
`max_wait_ticks=3`, and `reassignment_wait_ticks=8`.

## Running

```bash
python -m src.experiments.run_safety_ablation_experiment \
  --suite configs/experiment_suites/safety_guarantor_ablation.yaml \
  --backend mock \
  --output-dir results/safety_guarantor_ablation_mock
```

The default suite selects 10 scenarios each from `nav_easy`, `recovery_medium`,
and `pickplace_hard`, for 30 scenarios x 5 methods = 150 runs.

## Outputs

The runner writes:

- `raw.jsonl`
- `summary.csv`
- `summary.md`
- `traces/*.json`

Summary rows are produced at two levels: overall by method and by
`scenario_family + method`.

## Interpretation

The mock backend exists to validate the runner and artifact plumbing. It is not
evidence about real model quality. Expected trends are:

- LLM-only exposes more runtime conflicts and lower success.
- Static critic reduces invalid outputs but does not solve runtime conflicts.
- Centralized rules reduce motion, resource, and recovery conflicts.
- Repair improves success over one-shot generation.
- The full method adds approach-zone locking and should reduce pick/place
  manipulation-area risk.

DAG coordination is intentionally excluded from this suite and remains archived
as future work.
