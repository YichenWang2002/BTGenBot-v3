# SafetyGuarantor

`SafetyGuarantor` is the deterministic runtime layer that turns proposed
multi-robot actions into safe executions or waits. It wraps the existing
reservation table, resource manager, centralized rules, and approach-zone locks
behind one paper-facing abstraction.

## Not An LLM

The guarantor does not generate behavior trees, assignments, or repairs. The LLM
proposes robot-level BT actions. `SafetyGuarantor` checks those actions against
the current simulator state and either allows them or converts them to waits. It
is deterministic and auditable, so safety claims do not depend on model behavior.

## Safety Constraints

Motion safety covers grid movement. It uses cell reservations for vertex
conflicts and edge reservations for swap conflicts. These correspond to the
standard MAPF notions of two robots entering the same cell at the same timestep
or traversing the same edge in opposite directions.

Semantic resource safety covers shared task resources. Object, pickup-zone,
drop-zone, and recovery-zone locks prevent two robots from manipulating the same
semantic resource concurrently.

Manipulation-area safety covers the space around manipulation targets. Approach
zones use an `approach_radius` around pickup and drop cells, support passive
blocking of unrelated pass-through robots, and admit only task-relevant robots
into active manipulation areas.

## Metrics

Motion safety is reflected by `collision_event_count`, `actual_collision_count`,
`vertex_conflict_count`, `edge_conflict_count`, `rule_rejection_count`, and
`wait_count`.

Semantic resource safety is reflected by `resource_request_denied_count`,
`lock_wait_count`, `lock_wait_time`, `resource_conflict_count`, and
`rule_prevented_resource_conflict_count`.

Manipulation-area safety is reflected by `approach_zone_wait_count`,
`approach_zone_denied_count`, `rule_prevented_approach_conflict_count`,
`approach_lock_hold_time`, `approach_lock_starvation_count`, and
`approach_lock_reassignment_count`.

## Behavior Tree Execution

Behavior trees remain local robot controllers. The BT action stream proposes
navigation, pick/place, recovery, and resource actions. The centralized runtime
passes these proposals through `SafetyGuarantor`; allowed actions proceed, while
unsafe actions are represented as waits in the trace.

## MAPF Relationship

The motion layer is the MAPF-style part of the system. Cell reservations prevent
vertex conflicts, and edge reservations prevent edge-swap conflicts. The
guarantor extends this idea beyond MAPF by also checking semantic resources and
manipulation-area occupancy.

## Resource Locking

The semantic layer uses FIFO locks from `ResourceManager`. A robot that already
owns a resource may reacquire it, while other robots are queued and receive a
denial decision that the simulator converts to a wait.

## Approach-Zone Locking

Approach-zone locking treats pickup and drop neighborhoods as manipulation-area
resources. The zone is computed from the target cell and `approach_radius`.
Task-relevant robots may acquire the zone; pass-through or competing robots are
blocked while the lock is held. Starvation and reassignment policies remain in
the environment layer and are recorded as approach-lock metrics.

## Trace Fields

`rule_events` include the unified safety fields:

- `safety_layer`: `motion`, `semantic_resource`, `manipulation_area`, or `none`
- `conflict_type`: paper-facing conflict such as `cell_conflict`,
  `edge_conflict`, `object_lock_denied`, or `approach_zone_denied`
- `decision`: `allowed`, `denied`, or `converted_to_wait`
- `converted_to_wait`: whether the runtime replaced the action with a wait
- `resource_id`, `robot_id`, and `action_id`

DAG coordination is not part of this abstraction; it is archived as future work.
