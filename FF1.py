import pygame
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Set
from Maze import Maze
from config import CELL_SIZE, CELLS_W, CELLS_H, BG_COLOR, WALL_COLOR, GRID_COLOR, START_COLOR, GOAL_COLOR, PLAYER_COLOR

Cell = Tuple[int, int]

# Colors
WATER_COLOR = (80, 160, 255)      # flood expand (visited)
PATH_COLOR = (255, 215, 0)        # chosen path (gold)
BOT_COLOR = (255, 255, 255)       # bot color (white)

# Settings for path enumeration/selection
MAX_ENUM_SHORTEST = 3000          # cap for enumerating shortest paths
MAX_DISPLAY_PATHS = 24            # how many diverse shortest paths to overlay
SIMILARITY_MAX = 0.85             # Jaccard edge overlap threshold (skip near-duplicates)

# Bot speed
BOT_SPEED_CELLS_PER_SEC = 8.0

def random_interior_cell(maze: Maze) -> Cell:
    return (random.randint(1, maze.width - 2), random.randint(1, maze.height - 2))

def boundary_cells(maze: Maze) -> List[Cell]:
    w, h = maze.width, maze.height
    cells = []
    for x in range(w):
        cells.append((x, 0))
        cells.append((x, h - 1))
    for y in range(h):
        cells.append((0, y))
        cells.append((w - 1, y))
    # Deduplicate corners while preserving order
    return list(dict.fromkeys(cells))

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

def pick_hard_exit(maze: Maze, start: Cell) -> Cell:
    # Choose a boundary goal from a high-distance band (not strictly farthest)
    dist = bfs_distances(maze, start)
    candidates = [(c, dist.get(c, -1)) for c in boundary_cells(maze)]
    candidates = [(c, d) for c, d in candidates if d >= 0 and c != start]
    if not candidates:
        return random.choice(boundary_cells(maze))
    candidates.sort(key=lambda x: x[1])
    maxd = candidates[-1][1]
    threshold = int(maxd * 0.60)
    band = [c for c, d in candidates if d >= threshold]
    return random.choice(band) if band else candidates[-1][0]

def step_dir(a: Cell, b: Cell) -> str:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dx == 1 and dy == 0: return 'E'
    if dx == -1 and dy == 0: return 'W'
    if dx == 0 and dy == 1: return 'S'
    if dx == 0 and dy == -1: return 'N'
    return '?'

def path_metrics(path: List[Cell]) -> Tuple[int, int, int, List[str]]:
    tiles = len(path)
    if tiles <= 1:
        return tiles, 0, 0, []
    dirs = [step_dir(path[i], path[i + 1]) for i in range(tiles - 1)]
    turns = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
    # Longest straight run (in steps)
    max_stretch = 1 if dirs else 0
    run = 1
    for i in range(1, len(dirs)):
        if dirs[i] == dirs[i - 1]:
            run += 1
            max_stretch = max(max_stretch, run)
        else:
            run = 1
    return tiles, turns, max_stretch, dirs

def edge_set(path: List[Cell]) -> Set[Tuple[Cell, Cell]]:
    es = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        es.add((a, b) if a <= b else (b, a))  # undirected edge
    return es

def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def filter_diverse_paths(paths: List[List[Cell]], keep: int, sim_max: float) -> List[List[Cell]]:
    # Prefer fewer turns, then larger straight stretch, then lexicographic as fallback
    scored = []
    for p in paths:
        tiles, turns, ms, _ = path_metrics(p)
        scored.append((turns, -ms, p))
    scored.sort()
    kept: List[List[Cell]] = []
    kept_edges: List[Set] = []
    for _, __, p in scored:
        if len(kept) >= keep:
            break
        es = edge_set(p)
        if all(jaccard(es, ke) <= sim_max for ke in kept_edges):
            kept.append(p)
            kept_edges.append(es)
    # If we kept nothing (edge case), fall back to shortest by turns
    return kept if kept else [scored[0][2]] if scored else []

def enumerate_shortest_paths_from_parents(start: Cell, goal: Cell, parents: Dict[Cell, List[Cell]], cap: int) -> List[List[Cell]]:
    # parents maps node -> list of predecessor nodes at distance-1
    paths: List[List[Cell]] = []
    cur: List[Cell] = [goal]

    def dfs(c: Cell):
        if len(paths) >= cap:
            return
        if c == start:
            rev = cur[::-1]
            paths.append(rev)
            return
        for p in parents.get(c, []):
            cur.append(p)
            dfs(p)
            cur.pop()

    dfs(goal)
    return paths

def main():
    pygame.init()
    MARGIN = 16
    screen_w = CELLS_W * CELL_SIZE + MARGIN * 2
    screen_h = CELLS_H * CELL_SIZE + MARGIN * 2
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption('Maze (Flood Fill with Multiple Shortest Paths)')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Controls and tuning
    BFS_RATE_SLOW = 20
    BFS_RATE_FAST = 4000
    bfs_rate = BFS_RATE_SLOW
    AUTO_START = True

    # Difficulty presets (tuned to produce loops/branches; default hardest)
    complexity_levels = [
        dict(name="perfect", newest=0.95, loop=0.00, junction=0.00, braid=0.00),
        dict(name="hard_branches", newest=0.90, loop=0.14, junction=0.10, braid=0.02),
        dict(name="harder", newest=0.92, loop=0.22, junction=0.14, braid=0.03),
        dict(name="extreme", newest=0.94, loop=0.30, junction=0.20, braid=0.05),
        dict(name="insane", newest=0.95, loop=0.35, junction=0.25, braid=0.08),
    ]
    complexity_idx = 4  # hardest by default

    # State
    maze: Maze
    start: Cell
    goal: Cell
    level = complexity_levels[complexity_idx]

    # BFS state
    state = "idle"  # idle -> searching -> selecting -> traversing -> done
    dist: Dict[Cell, int] = {}
    parents: Dict[Cell, List[Cell]] = {}
    q: deque[Cell] = deque()
    frontier: Set[Cell] = set()
    found_goal_dist: Optional[int] = None

    # Paths
    all_shortest_paths: List[List[Cell]] = []
    display_paths: List[List[Cell]] = []
    path: List[Cell] = []
    path_set: Set[Cell] = set()
    highlight_idx: int = 0
    show_all_paths_overlay = True

    # Movement
    bot_cell_index = 0
    bot_px = bot_py = 0.0
    target_px = target_py = 0.0
    moving = False

    # Token bucket for BFS rate
    bfs_budget = 0.0

    def cell_center_px(cell: Cell) -> Tuple[float, float]:
        x, y = cell
        return (MARGIN + x * CELL_SIZE + CELL_SIZE / 2, MARGIN + y * CELL_SIZE + CELL_SIZE / 2)

    def set_caption(phase: str):
        title = (
            f"Flood Fill | {level['name']} | {CELLS_W}x{CELLS_H} | "
            f"loops={level['loop']:.2f} junction={level['junction']:.2f} braid={level['braid']:.2f} | "
            f"{phase.upper()}"
        )
        if state == "selecting" and display_paths:
            p = display_paths[highlight_idx]
            tiles, turns, ms, _ = path_metrics(p)
            title += f" | paths={len(display_paths)} | idx={highlight_idx+1}/{len(display_paths)} tiles={tiles} turns={turns} maxStraight={ms}"
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

        # Flood fill visited cells (only during searching)
        if state == "searching":
            pad = 5
            for (x, y) in dist.keys():
                rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                pygame.draw.rect(screen, WATER_COLOR, rect, border_radius=6)

        # Path overlays during selection/traversal
        pad = 5
        if state in ("selecting", "traversing", "done"):
            if state == "selecting" and show_all_paths_overlay:
                # Draw all (top N) shortest paths, each in a subtle color band
                # We draw cells, not polylines, for clarity
                colors = []
                n = max(1, len(display_paths))
                for i in range(n):
                    hue = i / n
                    # convert hue to RGB-ish palette
                    r = int(120 + 120 * (1 + -1 * abs(2 * hue - 1)))
                    g = int(100 + 120 * hue)
                    b = int(220 - 160 * hue)
                    colors.append((r % 256, g % 256, b % 256))
                for idx, p in enumerate(display_paths):
                    color = colors[idx]
                    for (x, y) in p:
                        rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                        pygame.draw.rect(screen, color, rect, border_radius=4)
            else:
                # Draw only the highlighted (selecting) or chosen (traversing/done) path
                draw_path = path if path else (display_paths[highlight_idx] if display_paths else [])
                for (x, y) in set(draw_path):
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
        if state in ("traversing", "done") and path:
            pygame.draw.circle(screen, BOT_COLOR, (int(bot_px), int(bot_py)), int(cs * 0.30))

        pygame.display.flip()

    def prepare_selection():
        nonlocal all_shortest_paths, display_paths, highlight_idx, state
        # Enumerate shortest paths from parents DAG
        all_shortest_paths = enumerate_shortest_paths_from_parents(start, goal, parents, MAX_ENUM_SHORTEST)
        # Filter for diversity (avoid trivially similar routes)
        display_paths = filter_diverse_paths(all_shortest_paths, MAX_DISPLAY_PATHS, SIMILARITY_MAX)
        highlight_idx = 0

        # Print to console
        print(f"\nShortest path count (enumerated, capped at {MAX_ENUM_SHORTEST}): {len(all_shortest_paths)}")
        print(f"Displaying top {len(display_paths)} diverse shortest paths:")
        print("idx | tiles | turns | maxStraight | directions (preview)")
        for idx, p in enumerate(display_paths):
            tiles, turns, ms, dirs = path_metrics(p)
            preview = ''.join(dirs)[:60] + ('…' if len(dirs) > 60 else '')
            print(f"{idx:3d} | {tiles:5d} | {turns:5d} | {ms:11d} | {preview}")

        state = "selecting"
        set_caption("selecting")

    def start_search():
        nonlocal state, dist, parents, q, frontier, found_goal_dist
        dist.clear()
        parents.clear()
        q.clear()
        frontier.clear()
        found_goal_dist = None

        q.append(start)
        dist[start] = 0
        frontier.add(start)
        state = "searching"
        set_caption("searching")

    def start_traversal(selected_path: List[Cell]):
        nonlocal path, path_set, bot_px, bot_py, target_px, target_py, moving, bot_cell_index, state
        if not selected_path:
            return
        path = selected_path
        path_set = set(path)
        bot_cell_index = 0
        bot_px, bot_py = cell_center_px(path[0])
        if len(path) > 1:
            target_px, target_py = cell_center_px(path[1])
            moving = True
        else:
            target_px, target_py = bot_px, bot_py
            moving = False
        state = "traversing"
        tiles, turns, ms, _ = path_metrics(path)
        print(f"\nChosen path: tiles={tiles}, turns={turns}, maxStraight={ms}")
        set_caption("traversing")

    def build_maze_and_positions():
        nonlocal maze, start, goal, level
        level = complexity_levels[complexity_idx]
        # Generate until at least 2 distinct shortest paths exist (genuine alternatives)
        while True:
            m = Maze.generate_growing_tree(CELLS_W, CELLS_H, newest_bias=level['newest'])
            m.add_loops_smart(loop_prob=level['loop'], avoid_dead_ends=True, center_bias=0.7)
            m.enrich_junctions(prob=level['junction'], avoid_dead_ends=True)
            m.braid_dead_ends(braid_prob=level['braid'])
            s = random_interior_cell(m)
            g = pick_hard_exit(m, s)

            # Quick check for multiple shortest paths
            # Run a fast BFS to build parents
            d: Dict[Cell, int] = {s: 0}
            par: Dict[Cell, List[Cell]] = {s: []}
            qq = deque([s])
            dist_g: Optional[int] = None
            while qq:
                c = qq.popleft()
                cd = d[c]
                if dist_g is not None and cd >= dist_g:
                    continue
                for n in m.neighbors(c):
                    nd = d.get(n)
                    if nd is None:
                        d[n] = cd + 1
                        par[n] = [c]
                        qq.append(n)
                        if n == g:
                            dist_g = cd + 1
                    elif nd == cd + 1:
                        # Extra parent (same shortest distance)
                        if c not in par.setdefault(n, []):
                            par[n].append(c)
            if g in d:
                paths_preview = enumerate_shortest_paths_from_parents(s, g, par, 3)
                if len(paths_preview) >= 2:
                    maze, start, goal = m, s, g
                    return

    # Initial kick-off
    build_maze_and_positions()
    if AUTO_START:
        start_search()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,):
                    running = False
                elif event.key == pygame.K_r:
                    build_maze_and_positions()
                    start_search()
                elif event.key == pygame.K_c:
                    complexity_idx = (complexity_idx + 1) % len(complexity_levels)
                    build_maze_and_positions()
                    start_search()
                elif event.key == pygame.K_f:
                    bfs_rate = 4000 if bfs_rate == BFS_RATE_SLOW else BFS_RATE_SLOW

                # Selection controls
                if state == "selecting" and display_paths:
                    if event.key in (pygame.K_RIGHT, pygame.K_d):
                        highlight_idx = (highlight_idx + 1) % len(display_paths)
                        set_caption("selecting")
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        highlight_idx = (highlight_idx - 1) % len(display_paths)
                        set_caption("selecting")
                    elif event.key == pygame.K_RETURN:
                        start_traversal(display_paths[highlight_idx])
                    elif event.key == pygame.K_s:
                        # pick min turns
                        best = min(range(len(display_paths)), key=lambda i: (path_metrics(display_paths[i])[1], -path_metrics(display_paths[i])[2]))
                        highlight_idx = best
                        set_caption("selecting")
                    elif event.key == pygame.K_l:
                        # pick max turns (more meandering)
                        best = max(range(len(display_paths)), key=lambda i: (path_metrics(display_paths[i])[1], path_metrics(display_paths[i])[2]))
                        highlight_idx = best
                        set_caption("selecting")
                    elif event.key == pygame.K_TAB or event.key == pygame.K_q:
                        show_all_paths_overlay = not show_all_paths_overlay
                        set_caption("selecting")

        # Update BFS (searching)
        if state == "searching":
            bfs_budget += bfs_rate * dt
            steps = int(bfs_budget)
            if steps > 0:
                bfs_budget -= steps

            done = False
            for _ in range(steps):
                if not q:
                    done = True
                    break

                c = q.popleft()
                frontier.discard(c)
                cd = dist[c]

                # If we already discovered goal distance, we only process nodes with depth < goal_dist
                if found_goal_dist is not None and cd >= found_goal_dist:
                    done = True
                    break

                for n in maze.neighbors(c):
                    newd = cd + 1
                    nd = dist.get(n)
                    if nd is None:
                        dist[n] = newd
                        parents[n] = [c]
                        q.append(n)
                        frontier.add(n)
                        if n == goal and found_goal_dist is None:
                            found_goal_dist = newd
                    elif newd == nd:
                        # Register extra parent for multiple shortest routes
                        if c not in parents.setdefault(n, []):
                            parents[n].append(c)

            if done or (goal in dist and found_goal_dist is not None):
                # Prepare selection of shortest paths
                prepare_selection()

        # Move bot along the path
        if state == "traversing" and path:
            speed_px = BOT_SPEED_CELLS_PER_SEC * CELL_SIZE
            dx = target_px - bot_px
            dy = target_py - bot_py
            distp = (dx * dx + dy * dy) ** 0.5
            if distp > 0:
                step = speed_px * dt
                if step >= distp:
                    bot_px, bot_py = target_px, target_py
                else:
                    bot_px += dx / distp * step
                    bot_py += dy / distp * step
            arrived = abs(bot_px - target_px) < 0.5 and abs(bot_py - target_py) < 0.5
            if arrived:
                bot_cell_index += 1
                if bot_cell_index >= len(path) - 1:
                    state = "done"
                    set_caption("done")
                else:
                    target_px, target_py = cell_center_px(path[bot_cell_index + 1])

        draw()

    pygame.quit()

if __name__ == '__main__':
    main()