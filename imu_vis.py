#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "bbos"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///

import curses
import time
import math
import numpy as np
from collections import deque
from bbos import Reader

def quat_to_euler(q):
    """Convert quaternion [w,x,y,z] to euler angles [roll,pitch,yaw]"""
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(np.clip(sinp, -1.0, 1.0))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def draw_bar(win, y, x, label, value, vmin, vmax, width=20, color_pair=0):
    """Draw a horizontal bar with label"""
    win.addstr(y, x, f"{label:>6}: ", curses.color_pair(color_pair))
    
    # Value display
    val_str = f"{value:7.2f}"
    win.addstr(y, x+9, val_str)
    
    # Bar visualization
    bar_x = x + 18
    win.addstr(y, bar_x-1, "[")
    win.addstr(y, bar_x+width, "]")
    
    # Normalize value to bar position
    norm_val = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    norm_val = np.clip(norm_val, 0, 1)
    bar_pos = int(norm_val * (width - 1))
    
    # Draw bar
    for i in range(width):
        if i == bar_pos:
            win.addstr(y, bar_x+i, "|", curses.color_pair(3))
        elif i == width // 2:
            win.addstr(y, bar_x+i, ".", curses.color_pair(1))
        else:
            win.addstr(y, bar_x+i, "-", curses.color_pair(1))

def draw_orientation_cube(win, y_start, x_start, roll, pitch, yaw):
    """Draw a simple 3D cube representation"""
    # Convert from radians to degrees for display
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    
    win.addstr(y_start, x_start, "Orientation:")
    win.addstr(y_start+1, x_start, f"Roll:  {roll_deg:6.1f} deg")
    win.addstr(y_start+2, x_start, f"Pitch: {pitch_deg:6.1f} deg")
    win.addstr(y_start+3, x_start, f"Yaw:   {yaw_deg:6.1f} deg")
    
    # Simple ASCII cube that rotates
    cube_y = y_start + 1
    cube_x = x_start + 20
    
    # Draw a simple representation based on orientation
    if abs(roll_deg) > 30:
        win.addstr(cube_y, cube_x, "  /\\  ", curses.color_pair(2))
        win.addstr(cube_y+1, cube_x, " /  \\ ", curses.color_pair(2))
        win.addstr(cube_y+2, cube_x, "/____\\", curses.color_pair(2))
    elif abs(pitch_deg) > 30:
        win.addstr(cube_y, cube_x, " ____ ", curses.color_pair(2))
        win.addstr(cube_y+1, cube_x, "|    |", curses.color_pair(2))
        win.addstr(cube_y+2, cube_x, "|____|", curses.color_pair(2))
    else:
        win.addstr(cube_y, cube_x, " ┌──┐ ", curses.color_pair(2))
        win.addstr(cube_y+1, cube_x, " │  │ ", curses.color_pair(2))
        win.addstr(cube_y+2, cube_x, " └──┘ ", curses.color_pair(2))

def main(stdscr):
    # Initialize curses
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(10)
    
    # Setup colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
    
    # Data history for graphs
    history_len = 50
    accel_history = [deque(maxlen=history_len) for _ in range(3)]
    gyro_history = [deque(maxlen=history_len) for _ in range(3)]
    temp_history = deque(maxlen=history_len)
    
    # Initialize readers
    r_data = Reader("imu.data")
    r_orient = Reader("imu.orientation")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Check for quit
        key = stdscr.getch()
        if key == ord('q') or key == ord('Q'):
            break
            
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        
        # Title
        title = "BracketBot IMU Visualizer"
        stdscr.addstr(0, (w-len(title))//2, title, curses.color_pair(2) | curses.A_BOLD)
        stdscr.addstr(1, (w-20)//2, "Press 'q' to quit", curses.color_pair(1))
        
        # Read IMU data
        if r_data.ready():
            accel = r_data.data['accel']
            gyro = r_data.data['gyro']
            temp = r_data.data['temp']
            
            # Update history
            for i in range(3):
                accel_history[i].append(accel[i])
                gyro_history[i].append(gyro[i])
            temp_history.append(temp)
            
            # Display accelerometer
            y_offset = 3
            stdscr.addstr(y_offset, 2, "Accelerometer (m/s^2):", curses.color_pair(2) | curses.A_BOLD)
            draw_bar(stdscr, y_offset+1, 4, "X", accel[0], -20, 20, 30, 1)
            draw_bar(stdscr, y_offset+2, 4, "Y", accel[1], -20, 20, 30, 1)
            draw_bar(stdscr, y_offset+3, 4, "Z", accel[2], -5, 25, 30, 1)
            
            # Display gyroscope
            y_offset = 8
            stdscr.addstr(y_offset, 2, "Gyroscope (rad/s):", curses.color_pair(2) | curses.A_BOLD)
            draw_bar(stdscr, y_offset+1, 4, "X", gyro[0], -5, 5, 30, 3)
            draw_bar(stdscr, y_offset+2, 4, "Y", gyro[1], -5, 5, 30, 3)
            draw_bar(stdscr, y_offset+3, 4, "Z", gyro[2], -5, 5, 30, 3)
            
            # Display temperature
            y_offset = 13
            stdscr.addstr(y_offset, 2, f"Temperature: {temp:.1f} C", curses.color_pair(2) | curses.A_BOLD)
            
            # Display magnitude
            accel_mag = np.linalg.norm(accel)
            gyro_mag = np.linalg.norm(gyro)
            stdscr.addstr(y_offset+1, 2, f"Accel magnitude: {accel_mag:.2f} m/s^2")
            stdscr.addstr(y_offset+2, 2, f"Gyro magnitude: {gyro_mag:.3f} rad/s")
        
        # Read orientation data
        if r_orient.ready():
            quaternion = r_orient.data['quaternion']
            
            # Display orientation
            y_offset = 17
            if quaternion is not None:
                # Convert quaternion to euler for display
                euler = quat_to_euler(quaternion)
                draw_orientation_cube(stdscr, y_offset, 2, euler[0], euler[1], euler[2])
        
        # Update stats
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Display stats
        stats_y = h - 2
        stdscr.addstr(stats_y, 2, f"FPS: {fps:.1f} | Frames: {frame_count}", curses.color_pair(1))
        
        stdscr.refresh()

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
