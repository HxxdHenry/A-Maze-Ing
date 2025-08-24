# CompareAll.py
import os
import sys
import time
import subprocess
import random

from config import CELLS_W, CELLS_H, CELL_SIZE

MARGIN = 16

def spawn(script, x, y, seed, rate, bot_speed, complexity_idx):
    env = os.environ.copy()
    env['MAZE_SEED'] = str(seed)                 # same maze in all windows
    env['COMPLEXITY_IDX'] = str(complexity_idx)  # same difficulty selection
    env['SEARCH_RATE'] = str(rate)               # same expansions/sec / prune/sec
    env['BOT_SPEED'] = str(bot_speed)            # same traversal speed (cells/sec)
    env['AUTO_START'] = '1'                      # auto-start searching
    env['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"     # window placement
    # Run each script in its own process/window
    return subprocess.Popen([sys.executable, script], env=env)

def main():
    # You can tweak these to your liking
    seed = int(time.time()) & 0xFFFFFFFF  # or set a fixed number like 123456
    complexity_idx = 2    # 0=perfect, 1=hard_branches, 2=harder, 3=extreme, 4=insane
    search_rate = 1500    # expansions/prunes per second
    bot_speed = 8.0       # cells per second for traversal

    w = CELLS_W * CELL_SIZE + 2 * MARGIN
    h = CELLS_H * CELL_SIZE + 2 * MARGIN
    pad = 20

    # Arrange windows (adjust if they overflow your screen)
    positions = [
        (pad, pad),
        (pad + w + pad, pad),
        (pad, pad + h + pad),
        (pad + w + pad, pad + h + pad),
        (pad, pad + 2 * (h + pad)),
    ]

    scripts = [
        "FloodFill.py",             # BFS
        "Dijkstra.py",
        "Astart.py",                # A*
        "Bidirectional_search.py",
        "Deadend.py",               # Dead-End Filling
    ]

    procs = []
    for (script, (x, y)) in zip(scripts, positions):
        p = spawn(script, x, y, seed, search_rate, bot_speed, complexity_idx)
        procs.append(p)

    # Keep the launcher alive until all windows close
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()

if __name__ == "__main__":
    main()