# Pygame Demo Guide

This project can render multi-robot navigation, recovery, and pickplace runs with
pygame. The demo entrypoint remains compatible with existing usage, and the new
render flags can override scenario YAML display settings.

## Force Render for `render: false`

Use `--render` to open a pygame window even when the scenario YAML has
`render: false`. Use `--headless` to force no window; `--headless` has higher
priority than `--render`.

```bash
python -m src.env.run_multi_demo \
  --scenario configs/generated/pickplace_hard/pickplace_hard_idx45_seed0_robots3.yaml \
  --centralized-rule true \
  --render \
  --fps 2 \
  --pause-on-end
```

Useful render flags:

- `--fps 2` slows the demo down for presentation.
- `--pause-on-end` keeps the final window open until ESC or window close.
- `--cell-size 48` controls grid cell size.

## Pickplace Success Demo

Use this case to show a centralized-rule pickplace run with object state,
pickup/drop markers, task progress, and resource lock visibility.

```bash
python -m src.env.run_multi_demo --scenario configs/generated/pickplace_hard/pickplace_hard_idx45_seed0_robots3.yaml --centralized-rule true --render --fps 2 --pause-on-end
```

## Pickplace Failure or Congestion Demo

Use this case to show congestion, wait behavior, collision cells, and timeout
state in the side panel.

```bash
python -m src.env.run_multi_demo --scenario configs/generated/pickplace_hard/pickplace_hard_idx13_seed0_robots3.yaml --centralized-rule true --render --fps 2 --pause-on-end
```

## Recovery Centralized Rule Demo

Recovery scenarios show centralized resource locks around recovery zones. The
side panel lists active wait queues when robots contend for the same resource.

```bash
python -m src.env.run_multi_demo \
  --scenario configs/generated/recovery_medium/recovery_medium_idx71_seed0_robots3.yaml \
  --centralized-rule true \
  --render \
  --fps 2 \
  --pause-on-end
```

## Replay a Trace

The replay tool reads a saved trace JSON and redraws each timestep without
rerunning the simulation.

```bash
python -m src.env.replay_trace --trace runs/demo_multi_robot_trace.json --fps 2 --pause-on-end
```

Controls:

- Space: pause or resume.
- Right: step one frame forward.
- Left: step one frame backward.
- ESC or window close: exit.

To export PNG frames:

```bash
python -m src.env.replay_trace \
  --trace runs/demo_multi_robot_trace.json \
  --fps 2 \
  --save-frames runs/demo_frames
```
