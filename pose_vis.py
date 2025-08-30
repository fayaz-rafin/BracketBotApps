#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = ["bbos"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///

import curses, time, math
from bbos import Reader

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    reader = Reader("localizer.pose")

    while True:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break

        stdscr.erase()
        stdscr.addstr(0, 0, "BracketBot XYΘ Display (press q to quit)", curses.A_BOLD)

        if reader.ready():
            row = reader.data
            try:
                deg = math.degrees(row['theta'])
                stdscr.addstr(2, 2, f"x: {row['x']:+.3f} m")
                stdscr.addstr(3, 2, f"y: {row['y']:+.3f} m")
                stdscr.addstr(4, 2, f"θ: {row['theta']:+.3f} rad ({deg:+.1f}°)")
            except Exception as e:
                stdscr.addstr(2, 2, f"Error reading row: {e}")
        else:
            stdscr.addstr(2, 2, "Waiting for data...")

        stdscr.refresh()
        time.sleep(0.05)

if __name__ == "__main__":
    curses.wrapper(main)
