import pygame
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Set
from Maze import Maze
from config import CELL_SIZE, CELLS_W, CELLS_H, BG_COLOR, WALL_COLOR, GRID_COLOR, START_COLOR, GOAL_COLOR, PLAYER_COLOR

Cell = Tuple[int, int]

# Extra colors (you can move these into config if you prefer)
WATER_COLOR = (80, 160, 255)      # flood fill (visited)
PATH_COLOR = (255, 215, 0)        # final shortest path (gold)
BOT_COLOR = (255, 255, 255)       # bot color (white)

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
    # Approximate maze diameter by double BFS
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

def main():
    pygame.init()
    MARGIN = 16
    screen_w = CELLS_W * CELL_SIZE + MARGIN * 2
    screen_h = CELLS_H * CELL_SIZE + MARGIN * 2
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption('Maze (Flood Fill Simulation)')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Controls and tuning
    # BFS rate is tokens-per-second (cells expanded per second). Toggle fast mode with F key.
    BFS_RATE_SLOW = 20
    BFS_RATE_FAST = 4000
    bfs_rate = BFS_RATE_SLOW

    BOT_SPEED_CELLS_PER_SEC = 8.0  # traversal speed
    AUTO_START = True              # start flood fill automatically on generate

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

    # Flood fill/BFS state
    state = "idle"  # "idle" -> "searching" -> "cleaning" -> "traversing" -> "done"
    visited: Set[Cell] = set()
    parent: Dict[Cell, Cell] = {}
    q: deque[Cell] = deque()
    frontier: Set[Cell] = set()
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
    bot_cell_index = 0  # index in path
    bot_px, bot_py = cell_center_px(start)
    target_px, target_py = bot_px, bot_py
    moving = False

    # Token bucket for BFS rate
    bfs_budget = 0.0

    def start_search():
        nonlocal state, visited, parent, q, frontier
        nonlocal search_started, search_start_time, search_end_time, search_time
        nonlocal path, path_set
        # Reset search
        visited.clear()
        parent.clear()
        q.clear()
        frontier.clear()
        path.clear()
        path_set.clear()

        q.append(start)
        visited.add(start)
        frontier.add(start)

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
            f"Flood Fill | {level['name']} | {CELLS_W}x{CELLS_H} | "
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

        # Flood fill visited cells (blue). If state>=cleaning and path_set is non-empty,
        # only draw the path (others are cleared).
        if state in ("searching", "cleaning", "traversing", "done"):
            draw_set: Set[Cell]
            if path_set:
                draw_set = path_set
                color = PATH_COLOR
            else:
                draw_set = visited
                color = WATER_COLOR

            pad = 5
            for (x, y) in draw_set:
                rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                pygame.draw.rect(screen, color, rect, border_radius=6)

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
        # # Search
        # if search_started and search_end_time == 0.0:
        #     cur = time.perf_counter() - search_start_time
        #     search_text = f"Search: {cur:.3f} s"
        # else:
        #     search_text = f"Search: {search_time:.3f} s"

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
        # hint = font.render("SPACE=start | R=regen | C=cycle difficulty | F=toggle BFS speed | ESC=quit", True, (200, 200, 200))
        # screen.blit(t1, (MARGIN, MARGIN - 8))
        # screen.blit(t2, (MARGIN + 200, MARGIN - 8))
        # screen.blit(t3, (MARGIN + 420, MARGIN - 8))
        # screen.blit(hint, (MARGIN, screen_h - MARGIN - 20))

        pygame.display.flip()

    def regenerate():
        nonlocal maze, start, goal, level, dist_sg
        nonlocal state, visited, parent, q, frontier, path, path_set
        nonlocal search_started, search_start_time, search_end_time, search_time
        nonlocal traverse_started, traverse_start_time, traverse_end_time, traverse_time
        nonlocal bot_px, bot_py, target_px, target_py, moving
        nonlocal bfs_budget  # reset BFS token bucket too

        maze, start, goal, level, dist_sg = build_maze_and_positions()

        # Reset search and traversal
        state = "idle"
        visited.clear()
        parent.clear()
        q.clear()
        frontier.clear()
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

        bfs_budget = 0.0
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
                    if bfs_rate == BFS_RATE_SLOW:
                        bfs_rate = BFS_RATE_FAST
                    else:
                        bfs_rate = BFS_RATE_SLOW

        # Update BFS (flood fill)
        if state == "searching":
            # token bucket for smooth, frame-independent rate
            bfs_budget += bfs_rate * dt
            steps = int(bfs_budget)
            if steps > 0:
                bfs_budget -= steps

            found = False
            for _ in range(steps):
                if not q:
                    break
                c = q.popleft()
                frontier.discard(c)
                for n in maze.neighbors(c):
                    if n not in visited:
                        visited.add(n)
                        parent[n] = c
                        q.append(n)
                        frontier.add(n)
                        if n == goal:
                            found = True
                            break
                if found:
                    break

            if found or (goal in visited):
                # finalise search timing
                search_end_time = time.perf_counter()
                search_time = search_end_time - search_start_time
                # build path
                p = reconstruct_path(parent, start, goal)
                if p:
                    path[:] = p
                    path_set = set(p)
                else:
                    # no path found (shouldn't happen in connected maze)
                    path[:] = []
                    path_set = set()
                # clear all non-path blue cells (keep only the correct path)
                visited.clear()
                visited |= path_set
                # move on to traversal
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
                        # end traversal timer
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