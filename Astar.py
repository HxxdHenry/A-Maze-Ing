import pygame
import random
import time
import heapq
from typing import Dict, List, Tuple, Optional, Set

# Adjust this import to the file that defines your Maze class
# e.g., from MazeManual import Maze  OR  from maze_manual import Maze
from Maze import Maze  # <-- CHANGE if your filename is different
from config import CELL_SIZE, CELLS_W, CELLS_H, BG_COLOR, WALL_COLOR, GRID_COLOR, START_COLOR, GOAL_COLOR

Cell = Tuple[int, int]

# Colors (you can move these into config if you want)
OPEN_COLOR = (80, 220, 140)   # open set (frontier)
CLOSED_COLOR = (80, 160, 255) # closed/expanded
PATH_COLOR = (255, 215, 0)    # final path (gold)
BOT_COLOR = (255, 255, 255)   # bot

def bfs_distances(maze: Maze, start: Cell) -> Dict[Cell, int]:
    from collections import deque
    dist = {start: 0}
    q = deque([start])
    while q:
        c = q.popleft()
        for n in maze.neighbors(c):
            if n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist

def farthest_cell(maze: Maze, start: Cell) -> Tuple[Cell, int]:
    dist = bfs_distances(maze, start)
    cell = max(dist, key=dist.get)
    return cell, dist[cell]

def choose_far_apart_pair(maze: Maze) -> Tuple[Cell, Cell, int]:
    start0 = (random.randrange(maze.width), random.randrange(maze.height))
    a, _ = farthest_cell(maze, start0)
    b, d = farthest_cell(maze, a)
    return a, b, d

def reconstruct_path(parent: Dict[Cell, Cell], start: Cell, goal: Cell) -> List[Cell]:
    if goal not in parent and goal != start:
        return []
    path = [goal]
    c = goal
    while c != start:
        c = parent[c]
        path.append(c)
    path.reverse()
    return path

def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def main():
    pygame.init()
    MARGIN = 16
    screen_w = CELLS_W * CELL_SIZE + MARGIN * 2
    screen_h = CELLS_H * CELL_SIZE + MARGIN * 2
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Maze (A* Simulation)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Difficulty presets (same vibe as your other script)
    complexity_levels = [
        dict(name="perfect", newest=0.95, loop=0.00, junction=0.00, braid=0.00),
        dict(name="hard_branches", newest=0.90, loop=0.12, junction=0.08, braid=0.01),
        dict(name="harder", newest=0.92, loop=0.20, junction=0.12, braid=0.01),
        dict(name="extreme", newest=0.94, loop=0.28, junction=0.18, braid=0.02),
        dict(name="insane", newest=0.95, loop=0.3, junction=0.25, braid=0.1),
    ]
    complexity_idx = 0  # default "extreme"

    # Search tuning
    EXP_RATE_SLOW = 250          # nodes expanded per second
    EXP_RATE_FAST = 5000
    exp_rate = EXP_RATE_SLOW
    HEURISTIC_WEIGHT = 1.0         # 1.0 = A*, 0.0 = Dijkstra
    AUTO_START = True

    BOT_SPEED_CELLS_PER_SEC = 8.0

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

    # State machine
    state = "idle"  # "idle" -> "searching" -> "cleaning" -> "traversing" -> "done"

    # A* search state
    open_heap: List[Tuple[float, int, Cell]] = []
    in_open: Set[Cell] = set()
    closed: Set[Cell] = set()
    g: Dict[Cell, float] = {}
    parent: Dict[Cell, Cell] = {}
    tie = 0  # tie-breaker counter to keep heap stable

    # Path and draw sets
    path: List[Cell] = []
    path_set: Set[Cell] = set()

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
    bot_idx = 0
    bot_px, bot_py = cell_center_px(start)
    target_px, target_py = bot_px, bot_py
    moving = False

    # Token bucket for expansions per second
    exp_budget = 0.0

    def set_caption(phase: str):
        title = (
            f"A* | {level['name']} | {CELLS_W}x{CELLS_H} | "
            f"newest={level['newest']:.2f} loops={level['loop']:.2f} "
            f"junction={level['junction']:.2f} braid={level['braid']:.2f} | "
            f"hW={HEURISTIC_WEIGHT:.2f} | start→goal={dist_sg} | {phase.upper()}"
        )
        pygame.display.set_caption(title)

    def start_search():
        nonlocal open_heap, in_open, closed, g, parent, tie
        nonlocal search_started, search_start_time, search_end_time, search_time
        nonlocal path, path_set, state

        # Reset search structures
        open_heap.clear()
        in_open.clear()
        closed.clear()
        g.clear()
        parent.clear()
        path.clear()
        path_set.clear()
        tie = 0

        # Seed A*
        g[start] = 0.0
        f0 = HEURISTIC_WEIGHT * manhattan(start, goal)
        heapq.heappush(open_heap, (f0, tie, start))
        in_open.add(start)

        search_started = True
        search_start_time = time.perf_counter()
        search_end_time = 0.0
        search_time = 0.0
        state = "searching"
        set_caption("searching")

    def start_traversal():
        nonlocal bot_idx, bot_px, bot_py, target_px, target_py, moving
        nonlocal traverse_started, traverse_start_time, traverse_end_time, traverse_time, state

        if not path:
            return
        bot_idx = 0
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

    def regenerate():
        nonlocal maze, start, goal, level, dist_sg
        nonlocal state, open_heap, in_open, closed, g, parent, path, path_set, tie
        nonlocal search_started, search_start_time, search_end_time, search_time
        nonlocal traverse_started, traverse_start_time, traverse_end_time, traverse_time
        nonlocal bot_px, bot_py, target_px, target_py, moving
        nonlocal exp_budget

        maze, start, goal, level, dist_sg = build_maze_and_positions()

        # Reset everything
        state = "idle"

        open_heap.clear()
        in_open.clear()
        closed.clear()
        g.clear()
        parent.clear()
        tie = 0

        path.clear()
        path_set.clear()

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

        exp_budget = 0.0
        set_caption("idle")

        if AUTO_START:
            start_search()

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

        # Search overlays
        pad = 5
        if state in ("searching",):
            # Closed (expanded) cells
            for (x, y) in closed:
                rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                pygame.draw.rect(screen, CLOSED_COLOR, rect, border_radius=6)
            # Open set (frontier)
            for (x, y) in in_open:
                if (x, y) in closed:
                    continue
                rect = pygame.Rect(m + x * cs + pad + 3, m + y * cs + pad + 3, cs - 2 * pad - 6, cs - 2 * pad - 6)
                pygame.draw.rect(screen, OPEN_COLOR, rect, border_radius=6)

        # After finding the path, show only the path
        if state in ("cleaning", "traversing", "done") and path_set:
            for (x, y) in path_set:
                rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                pygame.draw.rect(screen, PATH_COLOR, rect, border_radius=6)

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
        hint = font.render("SPACE=start | R=regen | C=cycle difficulty | F=toggle speed | H=toggle A*/Dijkstra | ESC=quit", True, (200, 200, 200))
        meta = font.render(f"exp/s={exp_rate}  hW={HEURISTIC_WEIGHT:.2f}", True, (180, 180, 180))
        screen.blit(t1, (MARGIN, MARGIN - 8))
        screen.blit(t2, (MARGIN + 200, MARGIN - 8))
        screen.blit(t3, (MARGIN + 420, MARGIN - 8))
        screen.blit(meta, (MARGIN, MARGIN + 14))
        screen.blit(hint, (MARGIN, screen_h - MARGIN - 20))

        pygame.display.flip()

    def finish_search(found: bool):
        nonlocal search_end_time, search_time, path, path_set, state
        search_end_time = time.perf_counter()
        search_time = search_end_time - search_start_time

        if found:
            p = reconstruct_path(parent, start, goal)
            path[:] = p
            path_set = set(p)
        else:
            path[:] = []
            path_set = set()

        # Clear other overlays so only path remains
        in_open.clear()
        closed.clear()
        state = "cleaning"

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

                elif event.key == pygame.K_r:
                    regenerate()

                elif event.key == pygame.K_c:
                    complexity_idx = (complexity_idx + 1) % len(complexity_levels)
                    regenerate()

                elif event.key == pygame.K_f:
                    exp_rate = EXP_RATE_FAST if exp_rate == EXP_RATE_SLOW else EXP_RATE_SLOW

                elif event.key == pygame.K_h:
                    # Toggle between A* (1.0) and Dijkstra (0.0)
                    HEURISTIC_WEIGHT = 0.0 if HEURISTIC_WEIGHT > 0.0 else 1.0
                    set_caption(state)

        # A* main loop (animated by token bucket)
        if state == "searching":
            exp_budget += exp_rate * dt
            steps = int(exp_budget)
            if steps > 0:
                exp_budget -= steps

            found = False
            for _ in range(steps):
                if not open_heap:
                    # No path
                    finish_search(False)
                    break

                f, _, current = heapq.heappop(open_heap)
                if current in closed:
                    continue  # stale entry
                if current in in_open:
                    in_open.discard(current)
                closed.add(current)

                if current == goal:
                    found = True
                    break

                # Expand neighbors
                for n in maze.neighbors(current):
                    # tentative g
                    tent_g = g[current] + 1.0
                    if n in closed and tent_g >= g.get(n, float("inf")):
                        continue
                    if tent_g < g.get(n, float("inf")):
                        parent[n] = current
                        g[n] = tent_g
                        tie += 1
                        f_n = tent_g + HEURISTIC_WEIGHT * manhattan(n, goal)
                        heapq.heappush(open_heap, (f_n, tie, n))
                        in_open.add(n)

            if found:
                finish_search(True)

        if state == "cleaning":
            # Transition immediately to traversal
            start_traversal()

        # Bot traversal
        if state == "traversing" and path:
            speed_px = BOT_SPEED_CELLS_PER_SEC * CELL_SIZE
            if moving:
                dx = target_px - bot_px
                dy = target_py - bot_py
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > 0:
                    step = speed_px * dt
                    if step >= dist:
                        bot_px, bot_py = target_px, target_py
                    else:
                        bot_px += dx / dist * step
                        bot_py += dy / dist * step

                arrived = abs(bot_px - target_px) < 0.5 and abs(bot_py - target_py) < 0.5
                if arrived:
                    bot_idx += 1
                    if bot_idx >= len(path) - 1:
                        moving = False
                        traverse_end_time = time.perf_counter()
                        traverse_time = traverse_end_time - traverse_start_time
                        state = "done"
                        set_caption("done")
                    else:
                        target_px, target_py = cell_center_px(path[bot_idx + 1])

        draw()

    pygame.quit()

if __name__ == "__main__":
    main()