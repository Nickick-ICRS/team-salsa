import numpy as np
from pathlib import Path

tempo = 120
counts = 8

duration = 60 * counts / tempo

#            1   &   2   &   3   &   4   &   5   &   6   &   7   &   8   &
l_16count = [1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0]
r_16count = [1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1]
d_16count = [1,  0, -1, -1,-.5,-.5,-.5,-.5, -1,  0,  1,  1, .5, .5, .5, .5]

foot_contacts = []
foot_xy_rel_hip = []
hip_orient_quat = []
hip_xy_pos = []
hip_height = []

foot_x = 0.3
foot_y = 0.2
hip_com_z = 0.5
hip_com_var = 0.05
hip_ang_var = 0.3

full = []

hip_x = 0
for i in range(16):
    l = l_16count[i]
    r = r_16count[i]
    nl = l_16count[(i+1) % 16]
    nr = l_16count[(i+1) % 16]
    d = d_16count[i]
    wd = (l + r + nl + nr) / (3 * sum([l, r]))

    hip_x += d * foot_x

    foot_contacts.append([l, r])
    foot_xy_rel_hip.append([[d*l*wd*foot_x, foot_y], [d*r*wd*foot_x, -foot_y]])
    hip_xy_pos.append([hip_x * wd, 0])
    hip_height.append(hip_com_z)
    theta = wd * hip_ang_var * (1 if l > r else -1)
    hip_orient_quat.append([np.array([np.sin(theta/2), 0, 0, np.cos(theta/2)])])
    full.append([
        l, r,
        d*l*wd*foot_x, foot_y, d*r*wd*foot_x, -foot_y,
        hip_x * wd, 0,
        hip_com_z,
        np.sin(theta/2), 0, 0, np.cos(theta/2)
    ])

# Save as numpy dict
data = {
    "foot_contacts": np.array(foot_contacts),                # (n_frames, 2)
    "foot_xy_rel_hip": np.array(foot_xy_rel_hip),            # (n_frames, 2, 2)
    "hip_orient_quat": np.array(hip_orient_quat),            # (n_frames, 4)
    "hip_xy_pos": np.array(hip_xy_pos),                      # (n_frames, 2)
    "hip_height": np.array(hip_height),                      # (n_frames,)
    "full_ordered": np.array(full),
}

output_path = Path(__file__).parent / 'generated/basic_step.npz'
np.savez_compressed(output_path, **data)
print(f"Saved reference data to {output_path}")
