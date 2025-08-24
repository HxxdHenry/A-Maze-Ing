import pygame
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Set
from Maze import Maze
from config import CELL_SIZE, CELLS_W, CELLS_H, BG_COLOR, WALL_COLOR, GRID_COLOR, START_COLOR, GOAL_COLOR

Cell = Tuple[int, int]

# Colors
FORWARD_COLOR = (80, 160, 255)    # forward flood (blue)
BACKWARD_COLOR = (160, 80, 255)   # backward flood (purple)
PATH_COLOR = (255, 215, 0)        # final path (gold)
BOT_COLOR = (255, 255, 255)       # bot color (white)

# Bot speed
BOT_SPEED_CELLS_PER_SEC = 8.0

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
    # Approximate diameter endpoints
    start0 = (random.randrange(maze.width), random.randrange(maze.height))
    a, _ = farthest_cell(maze, start0)
    b, d = farthest_cell(maze, a)
    return a, b, d

def reconstruct_bidirectional_path(
    parent_forward: Dict[Cell, Cell],
    parent_backward: Dict[Cell, Cell],
    start: Cell,
    goal: Cell,
    meet: Cell
) -> List[Cell]:
    path = []
    # start -> meet
    c = meet
    while c != start:
        path.append(c)
        c = parent_forward[c]
    path.append(start)
    path.reverse()
    # meet -> goal
    c = parent_backward[meet]
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
    pygame.display.set_caption('Bidirectional Flood Fill (Longest Single Path)')
    clock = pygame.time.Clock()

    # Controls
    SEARCH_RATE_SLOW = 10
    SEARCH_RATE_FAST = 4000
    search_rate = SEARCH_RATE_SLOW
    AUTO_START = True

    # Perfect maze (single-solution); newest≈1 for long corridors
    def build_perfect_longest_maze():
        m = Maze.generate_growing_tree(CELLS_W, CELLS_H, newest_bias=0.2)
        # No loops/junctions/braid => unique solution between any two cells
        s, g, dist_sg = choose_far_apart_pair(m)  # pick approximate diameter endpoints (longest path)
        return m, s, g, dist_sg

    maze, start, goal, dist_sg = build_perfect_longest_maze()

    # Bidirectional search state
    state = "idle"  # idle -> searching -> traversing -> done
    visited_forward: Set[Cell] = set()
    visited_backward: Set[Cell] = set()
    parent_forward: Dict[Cell, Cell] = {}
    parent_backward: Dict[Cell, Cell] = {}
    q_forward: deque[Cell] = deque()
    q_backward: deque[Cell] = deque()
    meet: Optional[Cell] = None
    path: List[Cell] = []

    # Movement
    bot_px = bot_py = 0.0
    target_px = target_py = 0.0
    bot_cell_index = 0
    moving = False

    # Token bucket
    search_budget = 0.0

    def cell_center_px(cell: Cell) -> Tuple[float, float]:
        x, y = cell
        return (MARGIN + x * CELL_SIZE + CELL_SIZE / 2, MARGIN + y * CELL_SIZE + CELL_SIZE / 2)

    def set_caption(phase: str):
        pygame.display.set_caption(
            f"Bidirectional (Perfect longest) | {CELLS_W}x{CELLS_H} | start→goal={dist_sg} | {phase.upper()}"
        )

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

        # Search visualization
        pad = 5
        if state in ("searching", "traversing", "done"):
            for (x, y) in visited_forward:
                rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                pygame.draw.rect(screen, FORWARD_COLOR, rect, border_radius=6)
            for (x, y) in visited_backward:
                rect = pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - 2 * pad, cs - 2 * pad)
                pygame.draw.rect(screen, BACKWARD_COLOR, rect, border_radius=6)

        # Final path
        if path:
            for (x, y) in set(path):
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

    def start_search():
        nonlocal state, visited_forward, visited_backward, parent_forward, parent_backward
        nonlocal q_forward, q_backward, meet, path
        visited_forward.clear()
        visited_backward.clear()
        parent_forward.clear()
        parent_backward.clear()
        q_forward.clear()
        q_backward.clear()
        meet = None
        path = []
        q_forward.append(start)
        visited_forward.add(start)
        parent_forward[start] = start
        q_backward.append(goal)
        visited_backward.add(goal)
        parent_backward[goal] = goal
        state = "searching"
        set_caption("searching")

    def start_traversal():
        nonlocal bot_px, bot_py, target_px, target_py, bot_cell_index, moving, state
        if not path:
            return
        bot_cell_index = 0
        bot_px, bot_py = cell_center_px(path[0])
        if len(path) > 1:
            target_px, target_py = cell_center_px(path[1])
            moving = True
        else:
            target_px, target_py = bot_px, bot_py
            moving = False
        state = "traversing"
        set_caption("traversing")

    # Initial kick-off
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
                    maze, start, goal, dist_sg = build_perfect_longest_maze()
                    start_search()
                elif event.key == pygame.K_f:
                    search_rate = SEARCH_RATE_FAST if search_rate == SEARCH_RATE_SLOW else SEARCH_RATE_SLOW

        # Update bidirectional search
        if state == "searching":
            search_budget += search_rate * dt
            steps = int(search_budget)
            if steps > 0:
                search_budget -= steps
            found = False
            meet_cell = None

            for _ in range(steps):
                # Forward step
                if q_forward and not found:
                    c = q_forward.popleft()
                    for n in maze.neighbors(c):
                        if n not in visited_forward:
                            visited_forward.add(n)
                            parent_forward[n] = c
                            q_forward.append(n)
                            if n in visited_backward:
                                found = True
                                meet_cell = n
                                break
                # Backward step
                if q_backward and not found:
                    c = q_backward.popleft()
                    for n in maze.neighbors(c):
                        if n not in visited_backward:
                            visited_backward.add(n)
                            parent_backward[n] = c
                            q_backward.append(n)
                            if n in visited_forward:
                                found = True
                                meet_cell = n
                                break
                if found:
                    break

            if found:
                meet = meet_cell
                path = reconstruct_bidirectional_path(parent_forward, parent_backward, start, goal, meet)
                state = "traversing"
                start_traversal()

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