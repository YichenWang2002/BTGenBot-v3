# DAG Coordination

The task DAG layer represents logical ordering constraints between high-level
tasks. It is independent of the simulator's centralized rule layer.

## What The DAG Represents

A `TaskDAG` contains task nodes such as navigation, pick, place, recovery,
assembly, and final-check tasks. Directed dependency edges say when one task may
start relative to another task. The initial dependency type is
`finish_to_start`, meaning the source task must complete before the target task
can run.

Typical constraints include:

- pick an object before placing that object
- complete several place tasks before assembly
- finish a recovery or clear task before dependent navigation or manipulation

## Difference From Centralized Rules

The DAG is a task-precondition layer. It decides whether a task is logically
ready to run.

The centralized rule layer is a runtime coordination layer. It handles movement
reservations, resource locks, recovery locks, approach-zone locks, and other
conflict prevention while robots execute.

The two layers are complementary: the DAG prevents starting tasks too early, and
centralized rules arbitrate shared space/resources during execution.

## YAML Schema

Scenario YAML can optionally include a `task_dag` block:

```yaml
task_dag:
  tasks:
    - task_id: pick_obj_0
      task_type: pick
      object_id: obj_0
      assigned_robot: robot_0
    - task_id: place_obj_0
      task_type: place
      object_id: obj_0
      assigned_robot: robot_0
    - task_id: final_check
      task_type: final_check
      assigned_robot: robot_1
  dependencies:
    - source: pick_obj_0
      target: place_obj_0
      type: finish_to_start
      description: pick before place
    - source: place_obj_0
      target: final_check
      type: finish_to_start
```

## LLM-Generated BT Usage

Future LLM-generated behavior trees can use the DAG to decide which tasks are
eligible for emission or execution:

1. Parse the scenario `task_dag`.
2. Validate that dependencies are acyclic and reference known tasks.
3. Ask `ready_tasks()` for tasks whose predecessors are completed.
4. Mark tasks running/completed/failed as BT execution progresses.
5. Keep centralized rules enabled for motion and resource safety.

This lets the planner express task ordering separately from low-level
coordination policy.
