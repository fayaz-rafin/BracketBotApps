# /// script
# dependencies = [
#   "bbos",
#   "numpy",
#   "matplotlib",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/BracketBotOS", editable = true }
# ///
from bbos import Reader, Writer, Config, Type
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
CFG = Config("so101")
# Recording for 10 seconds
traj = np.zeros((10*1000//20,CFG.dof), dtype=np.float32) # ctrl period 20ms, 10s * 20ms = 200 samples
print(traj.shape)
traj_path = Path(__file__).parent / "traj.txt"
i = 0
if not traj_path.exists():
    print("Recording...", flush=True)
    with Writer("so101.torque", Type("so101_torque")) as w_torque:
        w_torque['enable'] = np.zeros(CFG.dof, dtype=np.bool_)
    with Reader("so101.state") as r_state:
        while i < traj.shape[0]:
            if r_state.ready():
                traj[i] = r_state.data['pos']
                i += 1
    traj.tofile(traj_path)
    # debug by saving plot to file of each joint
    for i in range(CFG.dof):
        plt.plot(traj[:,i])
        plt.savefig(Path(__file__).parent / f"traj_{i}.png")
        plt.close()
# Play back
print("Playing back...", flush=True)

with Writer("so101.torque", Type("so101_torque")) as w_torque:
    w_torque['enable'] = np.ones(CFG.dof, dtype=np.bool_)
    with Writer("so101.ctrl", Type("so101_ctrl")) as w_ctrl:
        traj = np.fromfile(traj_path, dtype=np.float32)
        traj = traj.reshape(-1, CFG.dof)
        print(traj.shape)
        for i in range(traj.shape[0]):
            w_ctrl['pos'] = traj[i]
    w_torque['enable'] = np.zeros(CFG.dof, dtype=np.bool_)
print("Done writing!")