"""Replay a multi-robot JSON trace with pygame."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROBOT_COLORS = [
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (255, 127, 14),
    (148, 103, 189),
    (23, 190, 207),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--fps", default=2, type=int)
    parser.add_argument("--pause-on-end", action="store_true")
    parser.add_argument("--save-frames", type=Path)
    parser.add_argument("--cell-size", default=48, type=int)
    args = parser.parse_args(argv)

    payload = _load_trace(args.trace)
    frames = payload["trace"]
    if not frames:
        raise ValueError(f"Trace has no frames: {args.trace}")

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.init()
    cell_size = max(8, int(args.cell_size))
    fps = max(1, int(args.fps))
    grid_width, grid_height = _infer_grid_size(frames)
    panel_width = 390
    screen = pygame.display.set_mode(
        (grid_width * cell_size + panel_width, max(grid_height * cell_size, 360))
    )
    pygame.display.set_caption(f"trace replay: {payload.get('scenario_name', args.trace.name)}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, max(18, cell_size // 2))
    small_font = pygame.font.Font(None, 18)

    if args.save_frames is not None:
        args.save_frames.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            _draw_frame(
                pygame,
                screen,
                font,
                small_font,
                payload,
                frame,
                index,
                len(frames),
                grid_width,
                grid_height,
                cell_size,
                panel_width,
            )
            pygame.image.save(screen, args.save_frames / f"frame_{index:04d}.png")

    index = 0
    paused = False
    running = True
    while running:
        _draw_frame(
            pygame,
            screen,
            font,
            small_font,
            payload,
            frames[index],
            index,
            len(frames),
            grid_width,
            grid_height,
            cell_size,
            panel_width,
        )
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    paused = True
                    index = min(index + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    paused = True
                    index = max(index - 1, 0)

        if not paused:
            if index < len(frames) - 1:
                index += 1
            elif not args.pause_on_end:
                running = False

        clock.tick(fps if not paused else 30)

    pygame.quit()
    return 0


def _load_trace(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("trace"), list):
        raise ValueError("Trace JSON must contain a top-level trace list")
    return payload


def _infer_grid_size(frames: list[dict[str, Any]]) -> tuple[int, int]:
    max_x = 0
    max_y = 0

    def visit(position: Any) -> None:
        nonlocal max_x, max_y
        if (
            isinstance(position, list)
            and len(position) == 2
            and all(isinstance(value, int) for value in position)
        ):
            max_x = max(max_x, int(position[0]))
            max_y = max(max_y, int(position[1]))

    for frame in frames:
        for position in frame.get("robot_positions", {}).values():
            visit(position)
        for obj in frame.get("object_states", {}).values():
            visit(obj.get("position"))
            visit(obj.get("target_position"))
        for task in frame.get("task_states", {}).values():
            visit(task.get("pickup_position"))
            visit(task.get("drop_position"))
        for collision in frame.get("collisions", []):
            visit(collision.get("position"))
            visit(collision.get("from_cell"))
            visit(collision.get("to_cell"))

    return max_x + 1, max_y + 1


def _draw_frame(
    pygame: Any,
    screen: Any,
    font: Any,
    small_font: Any,
    payload: dict[str, Any],
    frame: dict[str, Any],
    index: int,
    total_frames: int,
    grid_width: int,
    grid_height: int,
    cell: int,
    panel_width: int,
) -> None:
    grid_pixel_width = grid_width * cell
    screen.fill((245, 245, 245))
    for x in range(grid_width):
        for y in range(grid_height):
            rect = pygame.Rect(x * cell, y * cell, cell, cell)
            pygame.draw.rect(screen, (220, 220, 220), rect, 1)

    _draw_trace_collisions(pygame, screen, frame.get("collisions", []), cell)
    _draw_trace_markers(pygame, screen, frame.get("task_states", {}), cell)
    _draw_trace_objects(pygame, screen, small_font, frame, cell)
    _draw_trace_robots(pygame, screen, font, frame.get("robot_positions", {}), cell)
    _draw_trace_panel(
        pygame,
        screen,
        small_font,
        payload,
        frame,
        index,
        total_frames,
        grid_pixel_width,
        panel_width,
    )


def _draw_trace_collisions(
    pygame: Any, screen: Any, collisions: list[dict[str, Any]], cell: int
) -> None:
    for collision in collisions:
        if "position" in collision:
            x, y = collision["position"]
            rect = pygame.Rect(x * cell, y * cell, cell, cell)
            pygame.draw.rect(screen, (255, 205, 205), rect)
            pygame.draw.rect(screen, (210, 40, 40), rect, max(2, cell // 12))
        elif "from_cell" in collision and "to_cell" in collision:
            from_x, from_y = collision["from_cell"]
            to_x, to_y = collision["to_cell"]
            start = (from_x * cell + cell // 2, from_y * cell + cell // 2)
            end = (to_x * cell + cell // 2, to_y * cell + cell // 2)
            pygame.draw.line(screen, (210, 40, 40), start, end, max(3, cell // 10))


def _draw_trace_markers(
    pygame: Any, screen: Any, task_states: dict[str, Any], cell: int
) -> None:
    for task in task_states.values():
        pickup = task.get("pickup_position")
        drop = task.get("drop_position")
        if pickup is not None:
            x, y = pickup
            rect = pygame.Rect(
                x * cell + cell // 5,
                y * cell + cell // 5,
                max(6, cell * 3 // 5),
                max(6, cell * 3 // 5),
            )
            pygame.draw.rect(screen, (65, 105, 225), rect, 2)
        if drop is not None:
            x, y = drop
            center = (x * cell + cell // 2, y * cell + cell // 2)
            pygame.draw.circle(screen, (20, 120, 80), center, max(6, cell // 3), 2)


def _draw_trace_objects(
    pygame: Any, screen: Any, font: Any, frame: dict[str, Any], cell: int
) -> None:
    robot_positions = frame.get("robot_positions", {})
    for object_id, obj in sorted(frame.get("object_states", {}).items()):
        status = str(obj.get("status", "available"))
        position = obj.get("position")
        held_by = obj.get("held_by")
        if position is None and held_by in robot_positions:
            robot_position = robot_positions[held_by]
            center = (
                robot_position[0] * cell + cell * 3 // 4,
                robot_position[1] * cell + cell // 4,
            )
        elif position is not None:
            center = (position[0] * cell + cell // 2, position[1] * cell + cell // 2)
        else:
            continue
        radius = max(5, cell // 6)
        pygame.draw.circle(screen, _object_color(status), center, radius)
        pygame.draw.circle(screen, (35, 35, 35), center, radius, 1)
        label = font.render(object_id.split("_")[-1], True, (20, 20, 20))
        screen.blit(label, label.get_rect(center=(center[0], center[1] + radius + 7)))


def _draw_trace_robots(
    pygame: Any, screen: Any, font: Any, robot_positions: dict[str, list[int]], cell: int
) -> None:
    for index, (robot_id, position) in enumerate(sorted(robot_positions.items())):
        color = ROBOT_COLORS[index % len(ROBOT_COLORS)]
        x, y = position
        center = (x * cell + cell // 2, y * cell + cell // 2)
        pygame.draw.circle(screen, color, center, max(8, cell // 3))
        label = font.render(robot_id.split("_")[-1], True, (255, 255, 255))
        screen.blit(label, label.get_rect(center=center))


def _draw_trace_panel(
    pygame: Any,
    screen: Any,
    font: Any,
    payload: dict[str, Any],
    frame: dict[str, Any],
    index: int,
    total_frames: int,
    panel_x: int,
    panel_width: int,
) -> None:
    panel_rect = pygame.Rect(panel_x, 0, panel_width, screen.get_height())
    pygame.draw.rect(screen, (250, 250, 250), panel_rect)
    pygame.draw.line(screen, (190, 190, 190), (panel_x, 0), (panel_x, screen.get_height()), 1)

    lines = _trace_panel_lines(payload, frame, index, total_frames)
    y = 12
    for line, color in lines:
        if y > screen.get_height() - 18:
            break
        text = font.render(line[:58], True, color)
        screen.blit(text, (panel_x + 12, y))
        y += 18


def _trace_panel_lines(
    payload: dict[str, Any], frame: dict[str, Any], index: int, total_frames: int
) -> list[tuple[str, tuple[int, int, int]]]:
    metrics = payload.get("metrics", {})
    normal = (35, 35, 35)
    muted = (95, 95, 95)
    alert = (190, 40, 40)
    status = "timeout" if metrics.get("timeout") else "success" if metrics.get("success") else "running"

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"scenario: {payload.get('scenario_name', 'trace')}", normal),
        (f"frame: {index + 1}/{total_frames}", normal),
        (f"timestep: {frame.get('timestep')}", normal),
        (f"status: {status}", normal),
        (f"centralized_rule_enabled: {metrics.get('centralized_rule_enabled', False)}", normal),
        ("", normal),
        ("robots", muted),
    ]
    statuses = frame.get("status", {})
    actions = frame.get("actions", {})
    for robot_id, position in sorted(frame.get("robot_positions", {}).items()):
        action_type = actions.get(robot_id, {}).get("action_type")
        suffix = f" action={action_type}" if action_type else ""
        lines.append((f"{robot_id} pos={position} status={statuses.get(robot_id)}{suffix}", normal))

    if frame.get("object_states"):
        task_by_object = {
            str(task.get("object_id")): task
            for task in frame.get("task_states", {}).values()
            if task.get("object_id") is not None
        }
        lines.append(("", normal))
        lines.append(("objects", muted))
        for object_id, obj in sorted(frame.get("object_states", {}).items()):
            task = task_by_object.get(object_id, {})
            held = f" held_by={obj.get('held_by')}" if obj.get("held_by") else ""
            lines.append((f"{object_id} status={obj.get('status')}{held}", normal))
            lines.append(
                (
                    f"  pickup={task.get('pickup_position')} drop={task.get('drop_position')}",
                    muted,
                )
            )

    lines.append(("", normal))
    lines.append(("resource lock wait queues", muted))
    wait_lines = _format_wait_queues(frame.get("resource_locks", {}).get("wait_queues", {}))
    if wait_lines:
        lines.extend((line, alert) for line in wait_lines)
    else:
        lines.append(("none", muted))

    lines.append(("", normal))
    lines.append(("collisions", muted))
    if frame.get("collisions"):
        lines.extend((_format_collision(collision), alert) for collision in frame["collisions"][:6])
    else:
        lines.append(("none", muted))
    return lines


def _format_wait_queues(wait_queues: dict[str, Any]) -> list[str]:
    lines = []
    for resource_id, queue in sorted(wait_queues.items()):
        queued = [
            str(item.get("robot_id", item)) if isinstance(item, dict) else str(item)
            for item in queue
        ]
        if queued:
            lines.append(f"{resource_id}: {', '.join(queued)}")
    return lines


def _format_collision(collision: dict[str, Any]) -> str:
    if "position" in collision:
        return (
            f"{collision.get('type')} cell={collision.get('position')} "
            f"robots={collision.get('robot_ids')}"
        )
    return (
        f"{collision.get('type')} edge={collision.get('from_cell')}->{collision.get('to_cell')} "
        f"robots={collision.get('robot_ids')}"
    )


def _object_color(status: str) -> tuple[int, int, int]:
    return {
        "available": (166, 110, 54),
        "held": (252, 186, 3),
        "placed": (70, 155, 105),
        "unavailable": (150, 150, 150),
    }.get(status, (166, 110, 54))


if __name__ == "__main__":
    raise SystemExit(main())
