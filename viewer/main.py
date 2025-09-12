# /// script
# dependencies = [
#   "bbos",
#   "rerun-sdk==0.24.0"
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
from bbos import Reader, Config
from bbos.tf import rot, trans
import numpy as np
import time
import rerun as rr
import socket
import uuid
from pathlib import Path
from bbos.time import Loop
HOSTNAME = socket.gethostname()


CFG_M = Config('mapping')
CFG_imu = Config('imu')

def main():
    rr.init("bracketbot-viewer", recording_id="bbos", default_blueprint=(Path(__file__).parent / "bracketbot-viewer.rbl").as_posix(), spawn=False)
    server_uri = rr.serve_grpc(grpc_port=9876, server_memory_limit="100MB")
    rr.serve_web_viewer(web_port=9090, connect_to=server_uri, open_browser=False)
    url = f"http://{HOSTNAME}.local:9090/?url=rerun%2Bhttp://{HOSTNAME}.local:9876/proxy"
    print("Viewer URL: ", url)
    with Reader("localizer.pose") as r_pose, \
         Reader("camera.points") as r_pts, \
         Reader("drive.ctrl") as r_ctrl, \
         Reader("imu.orientation") as r_ori, \
         Reader("mapping.voxels") as r_voxels, \
         Reader("camera.jpeg") as r_jpeg:
        #Loop.set_realtime(priority=90, cores={3})
        while True:
            if r_ori.ready() and False:
                rr.set_time("monotonic", timestamp=r_ori.data['timestamp'])
                rr.log("imu/orientation", rr.Scalars(r_ori.data['rpy']))
            if r_jpeg.ready() and False:
                rr.set_time("monotonic", timestamp=r_jpeg.data['timestamp'])
                rr.log("/camera", rr.EncodedImage(contents=r_jpeg.data['jpeg'],media_type="image/jpeg"))
            if r_pts.ready() and False:
                rr.set_time("monotonic", timestamp=r_pts.data['timestamp'])
                rr.log("/camera.points", rr.Points3D(r_pts.data['points'][:r_pts.data['num_points']], 
                                              colors=r_pts.data['colors'][:r_pts.data['num_points']]))
            if r_ctrl.ready() and False:
                for field in r_ctrl.data.dtype.names:
                    if field != 'timestamp':
                        rr.set_time("monotonic", timestamp=r_ctrl.data['timestamp'])
                        rr.log(f"drive/ctrl/{field}", rr.Scalars(r_ctrl.data[field]))
            if r_voxels.ready() and True:
                hits = r_voxels.data['keys'][(CFG_M.normalize(r_voxels.data['logodds']) > 0.75)]   # occupied voxels (hits)
                occ_voxels = CFG_M.unpack_keys(hits)
                # Filter out voxels below ground (z < 0)
                valid_mask = occ_voxels[:, 2] >= 0
                occ_voxels = occ_voxels[valid_mask]
                mask2d = (occ_voxels[:, 2] < 1) & (occ_voxels[:,2] >= 0.3)
                occ2d = occ_voxels[mask2d][:, :2]
                # Map height to color (blue to red)
                if len(occ_voxels) > 0:
                    heights = occ_voxels[:, 2]
                    h_min, h_max = heights.min(), heights.max()
                    if h_max > h_min:
                        normalized_heights = (heights - h_min) / (h_max - h_min)
                    else:
                        normalized_heights = np.zeros_like(heights)
                    
                    # Blue (0,0,255) to Red (255,0,0) based on height
                    colors = np.zeros((len(occ_voxels), 3), dtype=np.uint8)
                    colors[:, 0] = (normalized_heights * 255).astype(np.uint8)  # Red channel
                    colors[:, 2] = ((1 - normalized_heights) * 255).astype(np.uint8)  # Blue channel
                else:
                    colors = np.empty((0, 3), dtype=np.uint8)
                rr.set_time("monotonic", timestamp=r_voxels.data['timestamp'])
                rr.log("voxels", rr.Boxes3D(centers=occ_voxels, half_sizes=np.full_like(occ_voxels, CFG_M.voxel_size/2), colors=colors))
                rr.log("occ_grid/grid", rr.Points2D(occ2d))
            if r_pose.ready() and True:
                rr.set_time("monotonic", timestamp=r_pose.data['timestamp'])
                # Bot position (red) and direction indicator (blue)
                direction_length = 0.3  # meters ahead
                heading = (trans([r_pose.data['x'], r_pose.data['y'], 0]) @ rot([0, 0, 1], np.rad2deg(r_pose.data['theta'])))([0, direction_length, 0])
                points = np.array([[r_pose.data['x'], r_pose.data['y']], [heading[0], heading[1]]])
                colors = np.array([[255, 0, 0], [0, 0, 255]])
                rr.log("occ_grid/bot", rr.Points2D(points, colors=colors, radii=[0.05, 0.03]))
                rr.log("robot",
                    rr.Transform3D(
                        translation=[r_pose.data['x'], r_pose.data['y'], 0],
                        rotation_axis_angle=rr.RotationAxisAngle(
                            axis=[0, 0, 1],  # Z-axis for yaw rotation
                            radians=r_pose.data['theta']
                        ),
                    ),
                )
if __name__ == "__main__":
    main()
