import pygame
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import math
import time
from config import CELL_SIZE, CELLS_W, CELLS_H, BG_COLOR, WALL_COLOR, GRID_COLOR, START_COLOR, GOAL_COLOR, PLAYER_COLOR

Cell = Tuple[int, int]

DIRECTIONS = {
    'N': (0, -1),
    'S': (0, 1),
    'E': (1, 0),
    'W': (-1, 0),
}
DIR_ORDER = ['N', 'E', 'S', 'W']
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

# ----------------------------- Maze ------------------------------------------

@dataclass
class Maze:
    width: int
    height: int
    grid: List[List[Dict[str, bool]]] = field(init=False)

    def __post_init__(self):
        # grid[y][x] stores walls: True means wall exists
        self.grid = [[{d: True for d in DIRECTIONS} for _ in range(self.width)] for _ in range(self.height)]

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def carve(self, a: Cell, b: Cell):
        ax, ay = a
        bx, by = b
        if bx == ax and by == ay - 1:
            self.grid[ay][ax]['N'] = False
            self.grid[by][bx]['S'] = False
        elif bx == ax and by == ay + 1:
            self.grid[ay][ax]['S'] = False
            self.grid[by][bx]['N'] = False
        elif bx == ax + 1 and by == ay:
            self.grid[ay][ax]['E'] = False
            self.grid[by][bx]['W'] = False
        elif bx == ax - 1 and by == ay:
            self.grid[ay][ax]['W'] = False
            self.grid[by][bx]['E'] = False

    def is_open(self, cell: Cell, direction: str) -> bool:
        x, y = cell
        return not self.grid[y][x][direction]

    def neighbors(self, cell: Cell) -> List[Cell]:
        x, y = cell
        res = []
        for d, (dx, dy) in DIRECTIONS.items():
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < self.width and 0 <= ny_ < self.height and not self.grid[y][x][d]:
                res.append((nx_, ny_))
        return res

    def degree(self, cell: Cell) -> int:
        x, y = cell
        deg = 0
        for d in DIRECTIONS:
            if not self.grid[y][x][d]:
                deg += 1
        return deg

    @staticmethod
    def generate_growing_tree(width: int, height: int, newest_bias: float = 1.0, seed: Optional[int] = None) -> 'Maze':
        """
        Growing Tree algorithm:
        - newest_bias≈1.0 -> backtracker style (many dead ends, long corridors)
        - newest_bias≈0.0 -> Prim-like (more uniform branching, fewer dead ends)
        """
        if seed is not None:
            random.seed(seed)
        m = Maze(width, height)
        visited = [[False] * width for _ in range(height)]

        start = (random.randrange(width), random.randrange(height))
        visited[start[1]][start[0]] = True
        cells: List[Cell] = [start]

        while cells:
            # Choose index by bias
            if random.random() < newest_bias:
                i = len(cells) - 1  # newest
            else:
                i = random.randrange(len(cells))  # random from list

            x, y = cells[i]
            unvisited = []
            for (dx, dy) in DIRECTIONS.values():
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < width and 0 <= ny_ < height and not visited[ny_][nx_]:
                    unvisited.append((nx_, ny_))

            if unvisited:
                nx_, ny_ = random.choice(unvisited)
                m.carve((x, y), (nx_, ny_))
                visited[ny_][nx_] = True
                cells.append((nx_, ny_))
            else:
                cells.pop(i)
        return m

    def add_loops_smart(self, loop_prob: float, avoid_dead_ends: bool = True, center_bias: float = 0.5):
        """
        Add extra connections (loops) to increase branching factor.
        - avoid_dead_ends=True: avoid opening walls adjacent to degree-1 cells (preserves dead ends)
        - center_bias boosts probability near the center (harder)
        """
        if loop_prob <= 0:
            return
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        maxr = max(cx, cy) + 1e-9

        def center_weight(x: int, y: int) -> float:
            # 1 at edge, up to (1+center_bias) at center
            r = math.hypot((x - cx) / maxr, (y - cy) / maxr)
            return 1.0 + center_bias * (1.0 - min(r, 1.0))

        for y in range(self.height):
            for x in range(self.width):
                for d in ('E', 'S'):  # only process each wall once
                    if self.grid[y][x][d]:
                        dx, dy = DIRECTIONS[d]
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < self.width and 0 <= ny_ < self.height:
                            if avoid_dead_ends:
                                if self.degree((x, y)) <= 1 or self.degree((nx_, ny_)) <= 1:
                                    continue
                            p = loop_prob * center_weight(x, y)
                            if random.random() < p:
                                self.carve((x, y), (nx_, ny_))

    def enrich_junctions(self, prob: float, avoid_dead_ends: bool = True):
        """
        Turn some degree-2 cells into T-junctions (degree-3) to raise branching factor.
        Tries to connect to neighbors that are not dead ends (to keep dead-end count high).
        """
        if prob <= 0:
            return
        for y in range(self.height):
            for x in range(self.width):
                if self.degree((x, y)) == 2 and random.random() < prob:
                    closed_dirs = [d for d in DIR_ORDER if self.grid[y][x][d]]
                    random.shuffle(closed_dirs)
                    for d in closed_dirs:
                        dx, dy = DIRECTIONS[d]
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < self.width and 0 <= ny_ < self.height:
                            if avoid_dead_ends and self.degree((nx_, ny_)) <= 1:
                                continue
                            self.carve((x, y), (nx_, ny_))
                            break

    def braid_dead_ends(self, braid_prob: float):
        # Reduce some dead ends by carving one extra exit (adds loops)
        if braid_prob <= 0:
            return
        dead_ends = [(x, y) for y in range(self.height) for x in range(self.width)
                     if len(self.neighbors((x, y))) == 1]
        random.shuffle(dead_ends)
        for (x, y) in dead_ends:
            if random.random() < braid_prob:
                closed_dirs = [d for d in DIR_ORDER if self.grid[y][x][d]]
                random.shuffle(closed_dirs)
                for d in closed_dirs:
                    dx, dy = DIRECTIONS[d]
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < self.width and 0 <= ny_ < self.height:
                        self.carve((x, y), (nx_, ny_))
                        break

# ----------------------------- Helpers ----------------------------------------

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
    # Pick two cells far apart (approx maze diameter)
    start0 = (random.randrange(maze.width), random.randrange(maze.height))
    a, _ = farthest_cell(maze, start0)
    b, d = farthest_cell(maze, a)
    return a, b, d

def maze_metrics(maze: Maze) -> Tuple[int, int, float]:
    # dead_ends, branching_nodes (deg>=3), avg_degree
    degs = [maze.degree((x, y)) for y in range(maze.height) for x in range(maze.width)]
    dead_ends = sum(1 for d in degs if d == 1)
    branching = sum(1 for d in degs if d >= 3)
    avg_deg = sum(degs) / len(degs)
    return dead_ends, branching, avg_deg

# ----------------------- Pygame App (Manual + Smooth) -------------------------

def main():
    # Grid size (big = harder)
    MARGIN = 16

    # Player movement speed (cells per second)
    SPEED_CELLS_PER_SEC = 8.0

    # Difficulty presets
    # newest: bias for Growing-Tree (higher -> more dead ends)
    # loop: smart loop probability (higher -> more branching)
    # junction: probability to turn degree-2 cells into T-junctions (branching)
    # braid: small amount reduces only a few dead ends; keep low to retain many dead ends
    complexity_levels = [
        dict(name="perfect", newest=0.95, loop=0.00, junction=0.00, braid=0.00),
        dict(name="hard_branches", newest=0.90, loop=0.12, junction=0.08, braid=0.01),
        dict(name="harder", newest=0.92, loop=0.20, junction=0.12, braid=0.01),
        dict(name="extreme", newest=0.94, loop=0.28, junction=0.18, braid=0.02),
        dict(name="insane", newest=0.95, loop=0.3, junction=0.25, braid=0.1),
    ]
    # Default: significantly harder with many dead ends AND high branching
    complexity_idx = 0  # 0 - "insane"

    pygame.init()
    screen_w = CELLS_W * CELL_SIZE + MARGIN * 2
    screen_h = CELLS_H * CELL_SIZE + MARGIN * 2
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption('Maze (Manual Control)')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    def cell_center_px(cell: Cell) -> Tuple[int, int]:
        x, y = cell
        return int(MARGIN + x * CELL_SIZE + CELL_SIZE / 2), int(MARGIN + y * CELL_SIZE + CELL_SIZE / 2)

    def build_maze_and_positions():
        level = complexity_levels[complexity_idx]
        m = Maze.generate_growing_tree(CELLS_W, CELLS_H, newest_bias=level['newest'])
        # Boost branching without deleting most dead ends
        m.add_loops_smart(loop_prob=level['loop'], avoid_dead_ends=True, center_bias=0.6)
        m.enrich_junctions(prob=level['junction'], avoid_dead_ends=True)
        m.braid_dead_ends(braid_prob=level['braid'])  # keep tiny; preserves high dead-end count
        s, g, dist = choose_far_apart_pair(m)
        de, br, avgd = maze_metrics(m)
        return m, s, g, dist, (de, br, avgd), level

    maze, start, goal, dist_sg, (dead_ends, branch_nodes, avg_deg), level = build_maze_and_positions()

    # Player state
    player_cell: Cell = start
    player_px, player_py = cell_center_px(player_cell)
    moving = False
    move_dir: Optional[str] = None
    target_cell: Optional[Cell] = None
    target_px = target_py = 0
    queued_dir: Optional[str] = None
    won = False

    # Timer state
    timer_started = False
    timer_start_time = 0.0
    timer_end_time = 0.0
    timer_elapsed = 0.0

    def update_caption():
        title = (
            f"{level['name']} | {CELLS_W}x{CELLS_H} | "
            f"newest={level['newest']:.2f} loops={level['loop']:.2f} "
            f"junction={level['junction']:.2f} braid={level['braid']:.2f} | "
            f"DE={dead_ends} branches={branch_nodes} avgDeg={avg_deg:.2f} | "
            f"start→goal={dist_sg}"
        )
        if won:
            title += " | GOAL! Press R"
        pygame.display.set_caption(title)

    def can_move(cell: Cell, d: str) -> bool:
        return maze.is_open(cell, d)

    def begin_move(d: str):
        nonlocal moving, move_dir, target_cell, target_px, target_py, queued_dir
        if not can_move(player_cell, d):
            return
        dx, dy = DIRECTIONS[d]
        tc = (player_cell[0] + dx, player_cell[1] + dy)
        if not maze.in_bounds(tc):
            return
        tpx, tpy = cell_center_px(tc)
        moving = True
        move_dir = d
        target_cell = tc
        target_px, target_py = tpx, tpy
        if queued_dir == d:
            queued_dir = None

    def handle_idle_input():
        nonlocal queued_dir
        keys = pygame.key.get_pressed()

        def dir_pressed(d: str) -> bool:
            if d == 'N': return keys[pygame.K_UP] or keys[pygame.K_w]
            if d == 'S': return keys[pygame.K_DOWN] or keys[pygame.K_s]
            if d == 'W': return keys[pygame.K_LEFT] or keys[pygame.K_a]
            if d == 'E': return keys[pygame.K_RIGHT] or keys[pygame.K_d]
            return False

        if queued_dir and can_move(player_cell, queued_dir):
            begin_move(queued_dir)
            return

        for d in ['N', 'S', 'W', 'E']:
            if dir_pressed(d) and can_move(player_cell, d):
                begin_move(d)
                return

    def draw():
        screen.fill(BG_COLOR)
        cs = CELL_SIZE
        m = MARGIN

        # Grid (optional)
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

        # Start and Goal
        sx, sy = start
        gx, gy = goal
        srect = pygame.Rect(m + sx * cs + 6, m + sy * cs + 6, cs - 12, cs - 12)
        grect = pygame.Rect(m + gx * cs + 6, m + gy * cs + 6, cs - 12, cs - 12)
        pygame.draw.rect(screen, START_COLOR, srect)
        pygame.draw.rect(screen, GOAL_COLOR, grect)

        # Player
        pygame.draw.circle(screen, PLAYER_COLOR, (int(player_px), int(player_py)), int(cs * 0.35))

        # Timer display
        if timer_started:
            elapsed = time.perf_counter() - timer_start_time
        else:
            elapsed = timer_elapsed if won else 0.0
        timer_text = font.render(f"Time: {elapsed:.3f} s", True, (255, 255, 0))
        screen.blit(timer_text, (MARGIN, MARGIN - 8))

        # Win message
        if won:
            text = font.render("Goal reached! Press R to regenerate", True, (255, 255, 255))
            screen.blit(text, (m, screen_h - m - 24))

        pygame.display.flip()

    update_caption()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        speed_px = SPEED_CELLS_PER_SEC * CELL_SIZE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,):
                    running = False

                # Direction inputs (queue next turn)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    queued_dir = 'N'
                    if not moving:
                        begin_move('N')
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    queued_dir = 'S'
                    if not moving:
                        begin_move('S')
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    queued_dir = 'W'
                    if not moving:
                        begin_move('W')
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    queued_dir = 'E'
                    if not moving:
                        begin_move('E')

                elif event.key == pygame.K_r:
                    # Regenerate maze with current difficulty
                    maze, start, goal, dist_sg, (dead_ends, branch_nodes, avg_deg), level = build_maze_and_positions()
                    player_cell = start
                    player_px, player_py = cell_center_px(player_cell)
                    moving = False
                    move_dir = None
                    target_cell = None
                    queued_dir = None
                    won = False
                    # Reset timer
                    timer_started = False
                    timer_start_time = 0.0
                    timer_end_time = 0.0
                    timer_elapsed = 0.0
                    update_caption()

                elif event.key == pygame.K_c:
                    # Cycle difficulty level
                    complexity_idx = (complexity_idx + 1) % len(complexity_levels)
                    maze, start, goal, dist_sg, (dead_ends, branch_nodes, avg_deg), level = build_maze_and_positions()
                    player_cell = start
                    player_px, player_py = cell_center_px(player_cell)
                    moving = False
                    move_dir = None
                    target_cell = None
                    queued_dir = None
                    won = False
                    # Reset timer
                    timer_started = False
                    timer_start_time = 0.0
                    timer_end_time = 0.0
                    timer_elapsed = 0.0
                    update_caption()

        # Movement update (smooth sliding)
        if moving and target_cell is not None:
            if move_dir == 'N':
                player_py = max(player_py - speed_px * dt, target_py)
            elif move_dir == 'S':
                player_py = min(player_py + speed_px * dt, target_py)
            elif move_dir == 'W':
                player_px = max(player_px - speed_px * dt, target_px)
            elif move_dir == 'E':
                player_px = min(player_px + speed_px * dt, target_px)

            arrived = (abs(player_px - target_px) < 0.5) and (abs(player_py - target_py) < 0.5)
            if arrived:
                player_px, player_py = target_px, target_py
                prev_cell = player_cell
                player_cell = target_cell
                moving = False
                move_dir = None
                target_cell = None

                # Start timer when leaving start cell
                if not timer_started and prev_cell == start and player_cell != start:
                    timer_started = True
                    timer_start_time = time.perf_counter()

                # Check win
                if player_cell == goal:
                    won = True
                    if timer_started:
                        timer_end_time = time.perf_counter()
                        timer_elapsed = timer_end_time - timer_start_time
                        timer_started = False

                # Keep sliding if input held
                if not won:
                    handle_idle_input()

        elif not won:
            handle_idle_input()

        draw()

    pygame.quit()

if __name__ == '__main__':
    main()