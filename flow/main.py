#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
#   "fastapi",
#   "uvicorn",
#   "websockets",
#   "psutil",
#   "bbos",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///

import asyncio
import json
import os
import subprocess
import socket
import psutil
import numpy as np
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from bbos import Reader, Config
import threading
import queue
from datetime import datetime
from contextlib import ExitStack

# Configuration
CFG_SPKPN = Config("speakerphone")
AUDIO_BUFFER_MS = 5000  # 5 seconds of audio buffer

# Readers based on actual writers
READERS = [
    'camera.jpeg',
    'speakerphone.mic',
    'speakerphone.speaker',  # For speaker output visualization
    'led_strip.ctrl',
    'transcript',
    'drive.state',
    'drive.status',
    'camera.points'  # Point cloud data
]

# Daemon names for status checking
DAEMON_NAMES = ['camera', 'drive', 'led_strip', 'speakerphone', 'transcriber', 'depth']

# Global queues for writer data
queues = {}

# FastAPI app instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_writer_metadata() -> Dict[str, Any]:
    """Get metadata about active writers from unix sockets."""
    def get_data(sock: str):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        res = s.connect_ex(f'\0{sock}')
        s.settimeout(0.1)
        data = None
        if res == 0:
            try:
                data = s.recv(1024)
            except socket.timeout:
                pass
        s.close()
        return data

    # Use /proc/net/unix to find LISTEN state abstract sockets ending in .bbos
    awk_prog = 'NR>1 && $6=="01" && $NF ~ /^@.*\.bbos$/ {sub(/^@/, "", $NF); print $NF}'
    result = subprocess.run(["awk", awk_prog, "/proc/net/unix"], capture_output=True, text=True)
    sockets = result.stdout.splitlines()
    
    writer_metadata = {}
    
    for sock in sockets:
        w = sock.split("__")[0].replace(".bbos", "")
        if "timelog" not in sock:
            data = get_data(sock)
            if data:
                try:
                    info = json.loads(data)
                    writer_metadata[w] = {
                        'name': w,
                        'caller': info.get('caller', 'Unknown'),
                        'owner': info.get('owner', 'Unknown'),
                        'period': info.get('period', 0),
                        'dtype': info.get('dtype', [])
                    }
                except:
                    pass
    
    return writer_metadata

def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        load_avg = os.getloadavg()
        
        # Get top processes by CPU
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                pinfo = proc.info
                if pinfo['cpu_percent'] > 0:
                    processes.append({
                        'pid': pinfo['pid'],
                        'name': pinfo['name'],
                        'cpu': pinfo['cpu_percent']
                    })
            except:
                pass
        
        processes.sort(key=lambda x: x['cpu'], reverse=True)
        
        return {
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total,
                'used': memory.used,
                'percent': memory.percent
            },
            'swap': {
                'total': swap.total,
                'used': swap.used,
                'percent': swap.percent
            },
            'load_avg': load_avg,
            'top_processes': processes[:5]
        }
    except Exception as e:
        return {'error': str(e)}

def get_daemon_status(daemon_name: str) -> str:
    """Check if a daemon is running."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'daemon.py' in ' '.join(cmdline) and daemon_name in ' '.join(cmdline):
                    return 'running'
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return 'stopped'
    except Exception:
        return 'unknown'

def convert_numpy_to_json(data, skip_jpeg=False):
    """Convert numpy structured array to JSON-serializable dict."""
    json_data = {}
    
    if hasattr(data, 'dtype') and data.dtype.names:
        # It's a structured array
        for field_name in data.dtype.names:
            value = data[field_name]
            
            # Skip JPEG data if requested (for MJPEG streaming)
            if field_name == 'jpeg' and skip_jpeg:
                continue
            
            # Skip large point cloud arrays to avoid JSON bloat
            if field_name in ['points', 'colors'] and 'num_points' in data.dtype.names:
                # Don't send the actual arrays for point clouds
                continue
            
            if isinstance(value, np.ndarray):
                # For audio data, convert to int16 list (flatten if 2D)
                if field_name == 'audio' and value.dtype == np.int16:
                    if value.ndim == 2 and value.shape[1] == 1:
                        # Flatten mono audio from (samples, 1) to (samples,)
                        json_data[field_name] = value.flatten().tolist()
                    else:
                        json_data[field_name] = value.tolist()
                # For other arrays, convert based on type
                elif value.dtype in [np.float16, np.float32, np.float64]:
                    json_data[field_name] = value.tolist()
                elif value.dtype in [np.uint8, np.uint16, np.uint32]:
                    json_data[field_name] = value.tolist()
                else:
                    json_data[field_name] = value.tolist()
            elif isinstance(value, (np.int32, np.int64, np.uint32, np.uint64)):
                json_data[field_name] = int(value)
            elif isinstance(value, (np.float32, np.float64)):
                json_data[field_name] = float(value)
            elif isinstance(value, np.datetime64):
                json_data[field_name] = str(value)
            elif isinstance(value, bytes):
                json_data[field_name] = value.decode('utf-8', errors='replace')
            elif isinstance(value, np.bytes_):
                # Handle numpy bytes strings (like text from transcriber)
                json_data[field_name] = value.decode('utf-8', errors='replace').strip()
            else:
                json_data[field_name] = value
    else:
        # If it's already a dict, use it directly
        json_data = data if isinstance(data, dict) else {'data': data}
    
    return json_data

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML."""
    with open(os.path.join(os.path.dirname(__file__), 'frontend.html'), 'r') as f:
        return HTMLResponse(content=f.read())

@app.get("/api/writers")
async def get_writers():
    """Get metadata for all active writers."""
    return get_writer_metadata()

@app.get("/api/readers")
async def get_readers():
    """Get list of readers configured for this app."""
    return READERS

@app.get("/api/daemons")
async def get_daemons():
    """Get status of all daemons."""
    return {
        name: {
            'name': name,
            'status': get_daemon_status(name)
        }
        for name in DAEMON_NAMES
    }

@app.get("/api/system")
async def get_system():
    """Get system metrics."""
    return get_system_metrics()

@app.get("/api/pointcloud/status")
async def get_pointcloud_status():
    """Get current point cloud status."""
    if 'camera.points' in queues:
        try:
            # Try to get the latest data without blocking
            data = None
            # Drain the queue and keep only the latest
            while True:
                try:
                    data = queues['camera.points'].get_nowait()
                except queue.Empty:
                    break
            
            if data is not None:
                # Put it back for the binary websocket
                try:
                    queues['camera.points'].put_nowait(data)
                except queue.Full:
                    pass
                
                return {
                    'num_points': int(data['num_points']),
                    'timestamp': str(data['timestamp']) if 'timestamp' in data.dtype.names else None
                }
        except Exception as e:
            print(f"Error getting point cloud status: {e}")
    
    return {'num_points': 0}

@app.get("/mjpeg/camera")
async def mjpeg_stream():
    """Stream MJPEG video from camera."""
    headers = {"Content-Type": "multipart/x-mixed-replace; boundary=frame"}
    async def generate():
        while True:
            try:
                # Get the latest JPEG frame from the camera queue
                if 'camera.jpeg' in queues:
                    try:
                        data = queues['camera.jpeg'].get_nowait()
                        # Access numpy structured array fields
                        size = int(data["bytesused"])
                        jpeg = memoryview(data["jpeg"])[:size]
                        yield (b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: %d\r\n\r\n" % size + jpeg +
                            b"\r\n")
                    except queue.Empty:
                        pass
                
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in MJPEG stream: {e}")
                break
    
    return StreamingResponse(generate(), 
                           headers=headers)

@app.websocket("/ws/writer/{writer_name}")
async def writer_websocket(websocket: WebSocket, writer_name: str):
    """WebSocket endpoint for streaming writer data."""
    await websocket.accept()
    
    if writer_name not in queues:
        await websocket.close()
        return
    
    try:
        while True:
            # Get data from queue
            try:
                data = queues[writer_name].get_nowait()
                # Skip JPEG data for camera.jpeg (use MJPEG stream instead)
                skip_jpeg = (writer_name == 'camera.jpeg')
                json_data = convert_numpy_to_json(data, skip_jpeg=skip_jpeg)
                await websocket.send_json({
                    'writer': writer_name,
                    'data': json_data,
                    'timestamp': str(json_data.get('timestamp', datetime.now().isoformat()))
                })
            except queue.Empty:
                pass
            
            await asyncio.sleep(0.01)  # Small delay to prevent busy loop
                
    except WebSocketDisconnect:
        pass  # Normal disconnect
    except Exception as e:
        print(f"Error in WebSocket for {writer_name}: {e}")

@app.websocket("/ws/binary/camera.points")
async def points_binary_websocket(websocket: WebSocket):
    """Binary WebSocket endpoint for point cloud data."""
    print("Binary WebSocket connection attempt for camera.points")
    await websocket.accept()
    print("Binary WebSocket connection accepted for camera.points")
    
    try:
        while True:
            # Try to get data without blocking
            data = None
            try:
                # Get latest data from main loop
                if 'camera.points' in queues:
                    data = queues['camera.points'].get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            except Exception as e:
                print(f"Error getting data from queue: {e}")
                await asyncio.sleep(0.01)
                continue
                
            if data is not None:
                try:
                    # Pack binary data efficiently
                    num_points = int(data['num_points'])
                    print(f"Binary WS: Sending {num_points} points")
                    if num_points > 0 and num_points < 100000:  # Sanity check
                        # Create a binary message with header + points + colors
                        # Header: 4 bytes (num_points as int32)
                        header = np.array([num_points],dtype=np.int32).tobytes()
                        # Points: num_points * 3 * 2 bytes (float16)
                        points_data = data['points'][:num_points].tobytes()
                        # Colors: num_points * 3 * 1 byte (uint8)
                        colors_data = data['colors'][:num_points].tobytes() if 'colors' in data.dtype.names else b''
                        
                        # Send as binary message
                        await websocket.send_bytes(header + points_data + colors_data)
                        print(f"Binary WS: Sent {len(header + points_data + colors_data)} bytes")
                except Exception as e:
                    print(f"Error sending binary data: {e}")
                    
    except WebSocketDisconnect:
        print("Binary WebSocket disconnected")
    except Exception as e:
        print(f"Error in binary WebSocket for camera.points: {e}")

def ui(port: int, q: Dict[str, queue.Queue]):
    """Run the FastAPI application in a separate thread."""
    global queues
    queues = q
    uvicorn.run(app, host='0.0.0.0', port=port)

def main() -> None:
    """Entry point to run the Flow Dashboard server."""
    port = int(os.environ.get('FLOW_PORT', '8002'))
    
    # Create queues with appropriate sizes
    queues = {r: queue.Queue(maxsize=10 if r == 'camera.points' else 3) for r in READERS}
    readers = {r: Reader(r) for r in READERS}
    
    # Start UI thread
    ui_thread = threading.Thread(target=ui, args=(port, queues))
    ui_thread.daemon = True
    ui_thread.start()
    
    # Main reader loop
    with ExitStack() as stack:
        for reader in readers.values():
            stack.enter_context(reader)
            
        print(f"Flow Dashboard running on http://0.0.0.0:{port}")
        
        try:
            while True:
                for r in READERS:
                    if readers[r].ready():
                        try:
                            data = readers[r].data
                            if r == 'camera.points' and data is not None:
                                print(f"Got camera.points data: num_points={data['num_points']}")
                            # Put data in queue for all consumers
                            queues[r].put_nowait(data)
                        except queue.Full:
                            # Drop oldest data
                            try:
                                queues[r].get_nowait()
                            except queue.Empty:
                                pass
                            queues[r].put_nowait(data)
        except KeyboardInterrupt:
            print("\nShutting down...")

if __name__ == "__main__":
    main()