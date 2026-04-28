# Demo Cases

These fixed cases are for paper figures, group meetings, and screen recording.
Run all cases through the wrapper so the same scenario, rule setting, and render
speed are used consistently.

```bash
bash scripts/run_demo_case.sh pickplace_success_rule
```

Override playback speed when needed:

```bash
bash scripts/run_demo_case.sh pickplace_success_rule --fps 1
```

## Summary

| Case | Best use | Why |
| --- | --- | --- |
| `conflict_crossing_rule` | Paper screenshot | Small, clear example of rule-based collision avoidance. |
| `recovery_rule` | Paper screenshot or group meeting | Shows recovery-zone locking and waiting directly. |
| `pickplace_success_rule` | Paper screenshot and recording | Shows full 3-robot, 4-object pick/place success. |
| `pickplace_congested_rule` | Limitation slide | Resource rules work, but path congestion still causes timeout. |
| `conflict_crossing_no_rule` | Before/after comparison | Baseline conflict without centralized rule. |

## conflict_crossing_no_rule

Purpose: show path conflict when centralized rule is disabled.

Run:

```bash
bash scripts/run_demo_case.sh conflict_crossing_no_rule
```

Observe:

- Two robots try to enter the crossing cell.
- Collision cells are highlighted in the pygame view.
- The run times out because the conflict is never resolved.

Terminal metrics:

- `centralized_rule_enabled: False` confirms this is the no-rule baseline.
- `success: False`, `timeout: True` mean the task did not finish.
- `collision_count: 8` and `vertex_conflict_count: 8` are the main evidence.

## conflict_crossing_rule

Purpose: show how centralized rule avoids the same crossing conflict.

Run:

```bash
bash scripts/run_demo_case.sh conflict_crossing_rule
```

Observe:

- One robot is delayed while the other crosses.
- No collision cells should remain after rule arbitration.
- This is a compact case for a paper screenshot because the grid is small and the contrast with the no-rule case is obvious.

Terminal metrics:

- `centralized_rule_enabled: True` confirms the rule manager is active.
- `success: True`, `timeout: False` show the task completes.
- `collision_count: 0` is the key result.
- `rule_rejection_count: 1` and `wait_count: 1` show the rule prevented a conflicting move.

## recovery_rule

Purpose: show recovery-zone locking and waiting under centralized coordination.

Run:

```bash
bash scripts/run_demo_case.sh recovery_rule
```

Observe:

- Robots encounter blocked or temporary obstacle cells.
- The side panel shows recovery/resource wait queues when robots contend for the recovery zone.
- This is suitable for a paper screenshot when the lock/wait queue is visible.

Terminal metrics:

- `success: True` and `collision_count: 0` show recovery completes cleanly.
- `resource_request_denied_count: 3`, `lock_wait_count: 3`, and `wait_count: 3` show robots waited for a locked resource.
- `rule_prevented_resource_conflict_count: 3` is the main evidence that centralized recovery locking mattered.

## pickplace_success_rule

Purpose: show a successful 3-robot, 4-object pick/place demo with centralized resource control.

Run:

```bash
bash scripts/run_demo_case.sh pickplace_success_rule
```

Observe:

- Three robots move to pickup positions, carry objects, and place them at drop positions.
- Object colors/statuses change across available, held, placed, and unavailable states.
- Pickup/drop markers make the task structure visible.
- This is the best full-system paper screenshot and recording case.

Terminal metrics:

- `success: True`, `timeout: False` show the complete pick/place workflow succeeds.
- `collision_count: 0` confirms no path collision occurred.
- `resource_conflict_count: 0` confirms no unresolved object or zone resource conflict occurred.
- `resource_request_denied_count: 10` and `lock_wait_count: 10` are expected: robots waited instead of violating locks.

## pickplace_congested_rule

Purpose: show a limitation: centralized resource rules can prevent resource conflicts while path-level congestion still causes failure.

Run:

```bash
bash scripts/run_demo_case.sh pickplace_congested_rule
```

Observe:

- Robots make progress on some pick/place subtasks.
- The resource side remains controlled, but the path becomes congested.
- Collision highlights and repeated waiting make this the best limitation demo.

Terminal metrics:

- `centralized_rule_enabled: True` means this is not a no-rule baseline.
- `success: False` and `timeout: True` show the limitation.
- `collision_count: 171` and `rule_rejection_count: 173` indicate heavy path congestion.
- `resource_conflict_count: 0` shows resource rules remain effective despite the path-level failure.
