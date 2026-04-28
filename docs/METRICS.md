# Metrics

This project separates simulator-reported safety events from physical
trajectory conflicts reconstructed from traces. Paper-facing summaries should
prefer the new fields and treat historical `collision_count` as a compatibility
alias only.

## Collision Metrics

`collision_event_count` is the simulator-reported collision event count. It
includes direct collision events and blocked collision attempts reported by the
centralized safety layer. This is the clarified version of the historical
`collision_count` field.

`actual_collision_count` is computed by a trace-level audit. It counts physical
same-cell vertex conflicts plus edge-swap conflicts between adjacent timesteps.
Use this metric when claiming that the final executed robot trajectories have no
actual collision.

The two values can differ. A SafetyGuarantor rule may convert an unsafe move to
a wait and record a collision event or blocked attempt, while the executed trace
contains no physical collision. This is expected: `collision_event_count`
measures detected or prevented risk; `actual_collision_count` measures what
actually happened in the recorded trajectory.

Approach-zone locking primarily reduces `collision_event_count` by preventing
robots from entering manipulation areas at unsafe times. The final
`actual_collision_count` may already be zero if the safety layer converted those
attempts to waits.

`collision_count` is retained for backward compatibility with older result files
and scripts. It is deprecated and ambiguous for paper claims. New outputs also
include `collision_event_count_legacy`, which is numerically the same legacy
event value when present.

## SafetyGuarantor Metrics

Motion safety:

- `collision_event_count`
- `actual_collision_count`
- `vertex_conflict_event_count`
- `edge_conflict_event_count`
- `audited_vertex_conflict_count`
- `audited_edge_swap_conflict_count`
- `motion_wait_count`
- `rule_prevented_motion_conflict_count`

Semantic resource safety:

- `resource_conflict_count`
- `resource_request_denied_count`
- `lock_wait_count`
- `lock_wait_time`
- `rule_prevented_resource_conflict_count`
- `object_lock_denied_count`
- `pickup_zone_denied_count`
- `drop_zone_denied_count`
- `recovery_zone_denied_count`
- `recovery_conflict_count`

Manipulation-area safety:

- `approach_zone_wait_count`
- `approach_zone_denied_count`
- `approach_lock_hold_time`
- `approach_lock_starvation_count`
- `approach_lock_reassignment_count`
- `rule_prevented_approach_conflict_count`

## Paper Reporting

For paper tables, report both:

- `collision_event_count`
- `actual_collision_count`

`actual_collision_count` comes from trace-level audit. For saved runs, generate
the final-trace audit first:

```bash
python -m src.analysis.audit_trace_collisions \
  --runs-dir results/llm_pickplace_approach_v2_deepseek/runs \
  --final-only \
  --out results/llm_pickplace_approach_v2_deepseek/collision_audit_final.csv
```

Then pass that audit CSV to the paper table helper so existing summaries can
show trace-audited actual collisions without rerunning any LLM:

```bash
python -m src.visualization.paper_tables \
  --summary-csv results/llm_pickplace_approach_v2_deepseek/summary.csv \
  --collision-audit-csv results/llm_pickplace_approach_v2_deepseek/collision_audit_final.csv \
  --out results/llm_pickplace_approach_v2_deepseek/paper_table.md
```

The historical `collision_count` column should be treated as deprecated or
ambiguous in prose and captions.
