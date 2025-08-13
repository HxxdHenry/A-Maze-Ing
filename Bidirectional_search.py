import pygame
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Set
from Maze import Maze
from config import CELL_SIZE, CELLS_W, CELLS_H, BG_COLOR, WALL_COLOR, GRID_COLOR, START_COLOR, GOAL_COLOR, PLAYER_COLOR

Cell = Tuple[int, int]

# Extra colors (you can move these into config if you prefer)
FORWARD_COLOR = (80, 160, 255)    # forward search (blue)
BACKWARD_COLOR = (160, 80, 255)   # backward search (purple)
PATH_COLOR = (255, 215, 0)        # final shortest path (gold)
BOT_COLOR = (255, 255, 255)       # bot color (white)

def farthest_cell(maze: Maze, start: Cell, visited: Set[Cell]) -> Tuple[Cell, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        c = q.popleft()
        for n in maze.neighbors(c):
            if n not in dist and n not in visited:
                dist[n] = dist[c] + 1
                q.append(n)
    if not dist:
        return start, 0
    cell = max(dist, key=dist.get)
    return cell, dist[cell]

def choose_far_apart_pair(maze: Maze) -> Tuple[Cell, Cell, int]:
    # Approximate maze diameter by double BFS
    start0 = (random.randrange(maze.width), random.randrange(maze.height))
    a, _ = farthest_cell(maze, start0, set())
    b, d = farthest_cell(maze, a, set())
    return a, b, d

def reconstruct_bidirectional_path(
    parent_forward: Dict[Cell, Cell],
    parent_backward: Dict[Cell, Cell],
    start: Cell,
    goal: Cell,
    intersection: Cell
) -> List[Cell]:
    path = []
    # Forward path: start -> intersection
    c = intersection
    while c != start:
        path.append(c)
        c = parent_forward[c]
    path.append(start)
    path.reverse()
    # Backward path: intersection -> goal
    c = parent_backward[intersection]
    while c != goal:
        path.append(c)
        c = parent_backward[c]
    path.append(goal)
    return path

def main():
    pygame.init()
    MARGIN = 16
    screen_w = CELLS_W * CELL_SIZE + MARGIN * 2
    screen_h = CELLS_H * CELL_SIZE + MARGIN * 2
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption('Maze (Bidirectional Search Simulation)')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Controls and tuning
    # Search rate is tokens-per-second (cells expanded per second). Toggle fast mode with F key.
    SEARCH_RATE_SLOW = 500
    SEARCH_RATE_FAST = 4000
    search_rate = SEARCH_RATE_SLOW

    BOT_SPEED_CELLS_PER_SEC = 8.0  # traversal speed
    AUTO_START = True              # start search automatically on generate

    # Difficulty presets (same flavor as your manual script)
    complexity_levels = [
        dict(name="perfect", newest=0.95, loop=0.00, junction=0.00, braid=0.00),
        dict(name="hard_branches", newest=0.90, loop=0.12, junction=0.08, braid=0.01),
        dict(name="harder", newest=0.92, loop=0.20, junction=0.12, braid=0.01),
        dict(name="extreme", newest=0.94, loop=0.28, junction=0.18, braid=0.02),
        dict(name="insane", newest=0.95, loop=0.3, junction=0.25, braid=0.1),
    ]
    complexity_idx = 0  # default "extreme"

    def cell_center_px(cell: Cell) -> Tuple[float, float]:
        x, y = cell
        return (MARGIN + x * CELL_SIZE + CELL_SIZE / 2, MARGIN + y * CELL_SIZE + CELL_SIZE / 2)

    def build_maze_and_positions():
        level = complexity_levels[complexity_idx]
        m = Maze.generate_growing_tree(CELLS_W, CELLS_H, newest_bias=level['newest'])
        m.add_loops_smart(loop_prob=level['loop'], avoid_dead_ends=True, center_bias=0.6)
        m.enrich_junctions(prob=level['junction'], avoid_dead_ends=True)
        m.braid_dead_ends(braid_prob=level['braid'])
        s, g, dist = choose_far_apart_pair(m)
        return m, s, g, level, dist

    maze, start, goal, level, dist_sg = build_maze_and_positions()

    # Bidirectional search state
    state = "idle"  # "idle" -> "searching" -> "cleaning" -> "traversing" -> "done"
    visited_forward: Set[Cell] = set()
    visited_backward: Set[Cell] = set()
    parent_forward: Dict[Cell, Cell] = {}
    parent_backward: Dict[Cell, Cell] = {}
    q_forward: deque[Cell] = deque()
    q_backward: deque[Cell] = deque()
    frontier_forward: Set[Cell] = set()
    frontier_backward: Set[Cell] = set()
    path: List[Cell] = []
    path_set: Set[Cell] = set()
    intersection: Optional[Cell] = None

    # Timers
    search_started = False
    search_start_time = 0.0
    search_end_time = 0.0
    search_time = 0.0

    traverse_started = False
    traverse_start_time = 0.0
    traverse_end_time = 0.0
    traverse_time = 0.0

    # Movement along path
    bot_cell_index = 0  # index in path
    bot_px, bot_py = cell_center_px(start)
    target_px, target_py = bot_px, bot_py
    moving = False

    # Token bucket for search rate
    search_budget = 0.0

    def start_search():
        nonlocal state, visited_forward, visited_backward, parent_forward, parent_backward
        nonlocal q_forward, q_backward, frontier_forward, frontier_backward, intersection
        nonlocal search_started, search_start_time, search_end_time, search_time
        nonlocal path, path_set
        # Reset search
        visited_forward.clear()
        visited_backward.clear()
        parent_forward.clear()
        parent_backward.clear()
        q_forward.clear()
        q_backward.clear()
        frontier_forward.clear()
        frontier_backward.clear()
        path.clear()
        path_set.clear()
        intersection = None

        # Start forward search from start
        q_forward.append(start)
        visited_forward.add(start)
        frontier_forward.add(start)
        parent_forward[start] = start  # self-parent to mark start

        # Start backward search from goal
        q_backward.append(goal)
        visited_backward.add(goal)
        frontier_backward.add(goal)
        parent_backward[goal] = goal  # self-parent to mark goal

        search_started = True
        search_start_time = time.perf_counter()
        search_end_time = 0.0
        search_time = 0.0
        set_caption("searching")
        return

    def start_traversal():
        nonlocal state, bot_cell_index, bot_px, bot_py, target_px, target_py
        nonlocal traverse_started, traverse_start_time, traverse_end_time, traverse_time, moving
        if not path:
            return
        bot_cell_index = 0
        bot_px, bot_py = cell_center_px(path[0])
        if len(path) > 1:
            target_px, target_py = cell_center_px(path[1])
            moving = True
        else:
            moving = False
        traverse_started = True
        traverse_start_time = time.perf_counter()
        traverse_end_time = 0.0
        traverse_time = 0.0
        state = "traversing"
        set_caption("traversing")

    def set_caption(phase: str):
        title = (
            f"Bidirectional Search | {level['name']} | {CELLS_W}x{CELLS_H} | "
            f"newest={level['newest']:.2f} loops={level['loop']:.2f} "
            f"junction={level['junction']:.2f} braid={level['braid']:.2f} | "
            f"start→goal={dist_sg} | {phase.upper()}"
        )
        pygame.display.set_caption(title)

    def draw():
        screen.fill(BG_COLOR)
        cs = CELL_SIZE
        m = MARGIN

        # Grid
        for y in range(CELLS_H + 1):
            pygame.draw.line(screen, GRID_COLOR, (m, m + y * cs), (m + CELLS_W * cs, m + y * cs), 1)
        for x in range(CELLS_W + 1):
            pygame.draw.line(screen, GRID_COLOR, (m + x * cs, m), (m + x * cs, m + CELLS_H * cs), 1)

        # Walls
        for y in range(maze.height):
            for x in range(maze.width):
                cx = m + x * cs
                cy = m + y * cs
                walls = maze.grid[y][x]
                if walls['N']:
                    pygame.draw.line(screen, WALL_COLOR, (cx, cy), (cx + cs, cy), 3)
                if walls['S']:
                    pygame.draw.line(screen, WALL_COLOR, (cx, cy + cs), (cx + cs, cy + cs), 3)
                if walls['W']:
                    pygame.draw.line(screen, WALL_COLOR, (cx, cy), (cx, cy + cs), 3)
                if walls['E']:
                    pygame.draw.line(screen, WALL_COLOR, (cx + cs, cy), (cx + cs, cy + cs), 3)

        # Bidirectional search visited cells
        if state in ("searching", "cleaning", "traversing", "done"):
            pad = 5
            if path_set:
                # Draw only the final path in gold
                for (x, y) in path_set:
                    rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                    pygame.draw.rect(screen, PATH_COLOR, rect, border_radius=6)
            else:
                # Draw forward search in blue
                for (x, y) in visited_forward:
                    rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                    pygame.draw.rect(screen, FORWARD_COLOR, rect, border_radius=6)
                # Draw backward search in purple
                for (x, y) in visited_backward:
                    rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                    pygame.draw.rect(screen, BACKWARD_COLOR, rect, border_radius=6)

        # Start and Goal
        sx, sy = start
        gx, gy = goal
        srect = pygame.Rect(m + sx * cs + 6, m + sy * cs + 6, cs - 12, cs - 12)
        grect = pygame.Rect(m + gx * cs + 6, m + gy * cs + 6, cs - 12, cs - 12)
        pygame.draw.rect(screen, START_COLOR, srect, border_radius=4)
        pygame.draw.rect(screen, GOAL_COLOR, grect, border_radius=4)

        # Bot
        if state in ("traversing", "done"):
            pygame.draw.circle(screen, BOT_COLOR, (int(bot_px), int(bot_py)), int(cs * 0.30))

        # Timers
        # Search
        if search_started and search_end_time == 0.0:
            cur = time.perf_counter() - search_start_time
            search_text = f"Search: {cur:.3f} s"
        else:
            search_text = f"Search: {search_time:.3f} s"

        # Traverse
        if traverse_started and traverse_end_time == 0.0:
            cur = time.perf_counter() - traverse_start_time
            traverse_text = f"Traverse: {cur:.3f} s"
            total_text = f"Total: {(search_time + cur):.3f} s"
        else:
            traverse_text = f"Traverse: {traverse_time:.3f} s"
            total_text = f"Total: {(search_time + traverse_time):.3f} s"

        t1 = font.render(search_text, True, (255, 255, 0))
        t2 = font.render(traverse_text, True, (255, 255, 0))
        t3 = font.render(total_text, True, (255, 255, 255))
        hint = font.render("SPACE=start | R=regen | C=cycle difficulty | F=toggle search speed | ESC=quit", True, (200, 200, 200))
        screen.blit(t1, (MARGIN, MARGIN - 8))
        screen.blit(t2, (MARGIN + 200, MARGIN - 8))
        screen.blit(t3, (MARGIN + 420, MARGIN - 8))
        screen.blit(hint, (MARGIN, screen_h - MARGIN - 20))

        pygame.display.flip()

    def regenerate():
        nonlocal maze, start, goal, level, dist_sg
        nonlocal state, visited_forward, visited_backward, parent_forward, parent_backward
        nonlocal q_forward, q_backward, frontier_forward, frontier_backward, path, path_set, intersection
        nonlocal search_started, search_start_time, search_end_time, search_time
        nonlocal traverse_started, traverse_start_time, traverse_end_time, traverse_time
        nonlocal bot_px, bot_py, target_px, target_py, moving
        nonlocal search_budget  # reset search token bucket too

        maze, start, goal, level, dist_sg = build_maze_and_positions()

        # Reset search and traversal
        state = "idle"
        visited_forward.clear()
        visited_backward.clear()
        parent_forward.clear()
        parent_backward.clear()
        q_forward.clear()
        q_backward.clear()
        frontier_forward.clear()
        frontier_backward.clear()
        path.clear()
        path_set.clear()
        intersection = None

        search_started = False
        search_start_time = 0.0
        search_end_time = 0.0
        search_time = 0.0

        traverse_started = False
        traverse_start_time = 0.0
        traverse_end_time = 0.0
        traverse_time = 0.0

        bot_px, bot_py = cell_center_px(start)
        target_px, target_py = bot_px, bot_py
        moving = False

        search_budget = 0.0
        set_caption("idle")

        if AUTO_START:
            start_search()
            state = "searching"

    # Initial kick-off
    regenerate()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,):
                    running = False

                elif event.key == pygame.K_SPACE:
                    if state == "idle":
                        start_search()
                        state = "searching"

                elif event.key == pygame.K_r:
                    regenerate()

                elif event.key == pygame.K_c:
                    complexity_idx = (complexity_idx + 1) % len(complexity_levels)
                    regenerate()

                elif event.key == pygame.K_f:
                    if search_rate == SEARCH_RATE_SLOW:
                        search_rate = SEARCH_RATE_FAST
                    else:
                        search_rate = SEARCH_RATE_SLOW

        # Update bidirectional search
        if state == "searching":
            # Token bucket for smooth, frame-independent rate
            search_budget += search_rate * dt
            steps = int(search_budget)
            if steps > 0:
                search_budget -= steps

            found = False
            intersection_cell = None
            for _ in range(steps):
                # Alternate between forward and backward searches to keep them balanced
                # Forward step
                if q_forward and not found:
                    c = q_forward.popleft()
                    frontier_forward.discard(c)
                    for n in maze.neighbors(c):
                        if n not in visited_forward:
                            visited_forward.add(n)
                            parent_forward[n] = c
                            q_forward.append(n)
                            frontier_forward.add(n)
                            if n in visited_backward:
                                found = True
                                intersection_cell = n
                                break

                # Backward step
                if q_backward and not found:
                    c = q_backward.popleft()
                    frontier_backward.discard(c)
                    for n in maze.neighbors(c):
                        if n not in visited_backward:
                            visited_backward.add(n)
                            parent_backward[n] = c
                            q_backward.append(n)
                            frontier_backward.add(n)
                            if n in visited_forward:
                                found = True
                                intersection_cell = n
                                break

                if found:
                    break

            if found:
                # Finalize search timing
                search_end_time = time.perf_counter()
                search_time = search_end_time - search_start_time
                # Set intersection
                intersection = intersection_cell
                # Build path
                p = reconstruct_bidirectional_path(parent_forward, parent_backward, start, goal, intersection)
                if p:
                    path[:] = p
                    path_set = set(p)
                else:
                    # No path found (shouldn't happen in connected maze)
                    path[:] = []
                    path_set = set()
                # Clear all non-path cells (keep only the correct path)
                visited_forward.clear()
                visited_backward.clear()
                visited_forward |= path_set  # Use forward set to store path for drawing
                # Move on to traversal
                state = "cleaning"

        if state == "cleaning":
            # Small visual breath is optional; we can jump straight into traversal
            start_traversal()

        # Move bot along the path
        if state == "traversing" and path:
            speed_px = BOT_SPEED_CELLS_PER_SEC * CELL_SIZE
            if moving:
                # Move towards target
                dx = target_px - bot_px
                dy = target_py - bot_py
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 1e-6:
                    dist = 0.0
                if dist > 0:
                    step = speed_px * dt
                    if step >= dist:
                        bot_px, bot_py = target_px, target_py
                    else:
                        bot_px += dx / dist * step
                        bot_py += dy / dist * step

                arrived = abs(bot_px - target_px) < 0.5 and abs(bot_py - target_py) < 0.5
                if arrived:
                    # Advance to next cell
                    bot_cell_index += 1
                    if bot_cell_index >= len(path) - 1:
                        # Arrived at final cell
                        moving = False
                        # End traversal timer
                        traverse_end_time = time.perf_counter()
                        traverse_time = traverse_end_time - traverse_start_time
                        state = "done"
                        set_caption("done")
                    else:
                        target_px, target_py = cell_center_px(path[bot_cell_index + 1])

        draw()

    pygame.quit()

if __name__ == '__main__':
    main()