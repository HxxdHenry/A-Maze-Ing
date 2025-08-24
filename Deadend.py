import pygame
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Set

# Adjust this import to the file that defines your Maze class
# e.g., from MazeManual import Maze  OR  from maze_manual import Maze
from Maze import Maze  # <-- CHANGE if your filename is different
from config import CELL_SIZE, CELLS_W, CELLS_H, BG_COLOR, WALL_COLOR, GRID_COLOR, START_COLOR, GOAL_COLOR

Cell = Tuple[int, int]

# Colors
CORE_COLOR = (80, 200, 255)    # remaining core (after pruning)
PRUNED_COLOR = (240, 100, 100) # pruned dead-ends during animation
PATH_COLOR = (255, 215, 0)     # final shortest path
BOT_COLOR = (255, 255, 255)    # bot

def bfs_distances(maze: Maze, start: Cell) -> Dict[Cell, int]:
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

def build_adjacency(maze: Maze) -> Dict[Cell, Set[Cell]]:
    adj: Dict[Cell, Set[Cell]] = {}
    for y in range(maze.height):
        for x in range(maze.width):
            u = (x, y)
            adj[u] = set(maze.neighbors(u))
    return adj

def bfs_path_in_adj(adj: Dict[Cell, Set[Cell]], start: Cell, goal: Cell) -> List[Cell]:
    if start not in adj or goal not in adj:
        return []
    q = deque([start])
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    if goal not in parent:
        return []
    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path

def main():
    pygame.init()
    MARGIN = 16
    screen_w = CELLS_W * CELL_SIZE + MARGIN * 2
    screen_h = CELLS_H * CELL_SIZE + MARGIN * 2
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Maze (Dead-End Filling)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Difficulty presets (same vibe as your other scripts)
    complexity_levels = [
        dict(name="perfect", newest=0.95, loop=0.00, junction=0.00),
        dict(name="hard_branches", newest=0.90, loop=0.12, junction=0.08),
        dict(name="harder", newest=0.92, loop=0.20, junction=0.12),
        dict(name="extreme", newest=0.94, loop=0.28, junction=0.18),
        dict(name="insane", newest=0.95, loop=0.3, junction=0.25, braid=0.1),
    ]
    complexity_idx = 0  # default "extreme"

    # Animation tuning
    PRUNE_RATE_SLOW = 20  # cells pruned per second
    PRUNE_RATE_FAST = 5000
    prune_rate = PRUNE_RATE_SLOW
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
        s, g, dist = choose_far_apart_pair(m)
        return m, s, g, level, dist

    maze, start, goal, level, dist_sg = build_maze_and_positions()

    # State machine
    state = "idle"  # "idle" -> "pruning" -> "cleaning" -> "traversing" -> "done"

    # Dead-end fill state
    full_adj: Dict[Cell, Set[Cell]] = build_adjacency(maze)
    active_adj: Dict[Cell, Set[Cell]] = {u: set(vs) for u, vs in full_adj.items()}
    degree: Dict[Cell, int] = {u: len(vs) for u, vs in active_adj.items()}
    leaves: deque[Cell] = deque()
    pruned: Set[Cell] = set()
    core_set: Set[Cell] = set()

    # Path
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

    # Bot movement
    bot_idx = 0
    bot_px, bot_py = cell_center_px(start)
    target_px, target_py = bot_px, bot_py
    moving = False

    # Token bucket for pruning steps-per-second
    prune_budget = 0.0

    def set_caption(phase: str):
        title = (
            f"Dead-End Filling | {level['name']} | {CELLS_W}x{CELLS_H} | "
            f"newest={level['newest']:.2f} loops={level['loop']:.2f} "
            f"junction={level['junction']:.2f} | start→goal={dist_sg} | {phase.upper()}"
        )
        pygame.display.set_caption(title)

    def seed_leaves():
        nonlocal leaves
        leaves.clear()
        for u, d in degree.items():
            if d == 1 and u not in (start, goal):
                leaves.append(u)

    def start_search():
        nonlocal full_adj, active_adj, degree, leaves, pruned, core_set
        nonlocal search_started, search_start_time, search_end_time, search_time, state

        full_adj = build_adjacency(maze)
        active_adj = {u: set(vs) for u, vs in full_adj.items()}
        degree = {u: len(vs) for u, vs in active_adj.items()}
        pruned.clear()
        core_set.clear()

        seed_leaves()

        search_started = True
        search_start_time = time.perf_counter()
        search_end_time = 0.0
        search_time = 0.0
        state = "pruning"
        set_caption("pruning")

    def reachable(adj: Dict[Cell, Set[Cell]], src: Cell) -> Set[Cell]:
        if src not in adj:
            return set()
        seen = {src}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    def finish_pruning():
        nonlocal search_end_time, search_time, core_set, active_adj, path, path_set, state
        # End search timer
        search_end_time = time.perf_counter()
        search_time = search_end_time - search_start_time

        # Keep only nodes that are on some S->G path: intersection of reachability
        rs = reachable(active_adj, start)
        rg = reachable(active_adj, goal)
        keep = rs & rg
        core_set = set(keep)

        # Filter adjacency to S->G component
        active_adj = {u: set(v for v in vs if v in keep) for u, vs in active_adj.items() if u in keep}

        # Pick a shortest path within the remaining core
        path = bfs_path_in_adj(active_adj, start, goal)
        path_set = set(path)

        # Clear overlays except the path (match A*/Dijkstra style)
        state = "cleaning"

    def start_traversal():
        nonlocal bot_idx, bot_px, bot_py, target_px, target_py, moving
        nonlocal traverse_started, traverse_start_time, traverse_end_time, traverse_time, state

        if not path:
            state = "done"
            set_caption("done")
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
        nonlocal state, full_adj, active_adj, degree, leaves, pruned, core_set
        nonlocal path, path_set
        nonlocal search_started, search_start_time, search_end_time, search_time
        nonlocal traverse_started, traverse_start_time, traverse_end_time, traverse_time
        nonlocal bot_px, bot_py, target_px, target_py, moving, prune_budget

        maze, start, goal, level, dist_sg = build_maze_and_positions()

        full_adj = build_adjacency(maze)
        active_adj = {u: set(vs) for u, vs in full_adj.items()}
        degree = {u: len(vs) for u, vs in active_adj.items()}
        leaves = deque()
        pruned = set()
        core_set = set()

        path = []
        path_set = set()

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

        prune_budget = 0.0
        state = "idle"
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

        pad = 5

        # Pruning overlay (red for cells pruned so far), and core (cyan) while pruning
        if state == "pruning":
            for (x, y) in pruned:
                rect = pygame.Rect(m + x * cs + pad + 3, m + y * cs + pad + 3, cs - 2 * pad - 6, cs - 2 * pad - 6)
                pygame.draw.rect(screen, PRUNED_COLOR, rect, border_radius=6)
            current_core = [u for u, d in degree.items() if (d > 0 or u in (start, goal)) and u not in pruned]
            for (x, y) in current_core:
                rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                pygame.draw.rect(screen, CORE_COLOR, rect, border_radius=6)

        # After pruning, show only the path (like A*/Dijkstra)
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
        # # Search (prune)
        # if search_started and search_end_time == 0.0:
        #     cur = time.perf_counter() - search_start_time
        #     search_text = f"Prune: {cur:.3f} s"
        # else:
        #     search_text = f"Prune: {search_time:.3f} s"

        # # Traverse
        # if traverse_started and traverse_end_time == 0.0:
        #     cur = time.perf_counter() - traverse_start_time
        #     traverse_text = f"Traverse: {cur:.3f} s"
        #     total_text = f"Total: {(search_time + cur):.3f} s"
        # else:
        #     traverse_text = f"Traverse: {traverse_time:.3f} s"
        #     total_text = f"Total: {(search_time + traverse_time):.3f} s"

        # t1 = font.render(search_text, True, (255, 255, 0))
        # t2 = font.render(traverse_text, True, (255, 255, 0))
        # t3 = font.render(total_text, True, (255, 255, 255))
        # hint = font.render("SPACE=start | R=regen | C=cycle difficulty | F=toggle prune speed | ESC=quit", True, (200, 200, 200))
        # meta = font.render(f"prunes/s={prune_rate}", True, (180, 180, 180))
        # screen.blit(t1, (MARGIN, MARGIN - 8))
        # screen.blit(t2, (MARGIN + 200, MARGIN - 8))
        # screen.blit(t3, (MARGIN + 420, MARGIN - 8))
        # screen.blit(meta, (MARGIN, MARGIN + 14))
        # screen.blit(hint, (MARGIN, screen_h - MARGIN - 20))

        pygame.display.flip()

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
                    prune_rate = PRUNE_RATE_FAST if prune_rate == PRUNE_RATE_SLOW else PRUNE_RATE_SLOW

        # Dead-end filling (animated)
        if state == "pruning":
            prune_budget += prune_rate * dt
            steps = int(prune_budget)
            if steps > 0:
                prune_budget -= steps

            if not leaves and all((degree.get(u, 0) != 1 or u in (start, goal)) for u in degree):
                # No more prunable leaves
                finish_pruning()

            else:
                for _ in range(steps):
                    if not leaves:
                        break
                    u = leaves.popleft()
                    if u in pruned:
                        continue
                    if u in (start, goal):
                        continue
                    if degree.get(u, 0) != 1:
                        continue
                    # u must have exactly one neighbor v
                    if not active_adj[u]:
                        continue
                    v = next(iter(active_adj[u]))
                    # remove edge (u, v)
                    active_adj[u].discard(v)
                    active_adj[v].discard(u)
                    degree[u] = 0
                    degree[v] = max(0, degree.get(v, 0) - 1)
                    pruned.add(u)
                    # If v becomes a leaf and is not S/G, enqueue it
                    if v not in (start, goal) and degree[v] == 1:
                        leaves.append(v)

            # When pruning finishes (queue empty AND no 1-degree nodes), finalize
            if not leaves and all((degree.get(u, 0) != 1 or u in (start, goal)) for u in degree):
                finish_pruning()

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