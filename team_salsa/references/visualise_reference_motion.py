import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path


def visualise_reference_motion(reference_path):
    data = np.load(reference_path)
    foot_contacts = data["foot_contacts"]
    foot_xy_rel_hip = data["foot_xy_rel_hip"]  # (n_frames, 2, 2)
    hip_height = data["hip_height"]            # (n_frames,)
    n_frames = foot_xy_rel_hip.shape[0]

    # MJCF with 3 sites (visual spheres)
    mjcf = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <site name="hip_site" pos="0 0 1" size="0.04" rgba="0.2 0.2 1 1" type="sphere"/>
        <site name="left_foot_site" pos="0 0 0.8" size="0.03" rgba="1 0.2 0.2 1" type="sphere"/>
        <site name="right_foot_site" pos="0 0 0.8" size="0.03" rgba="0.2 1 0.2 1" type="sphere"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(mjcf)
    sim = mujoco.MjData(model)

    hip_site_id = model.site("hip_site").id
    left_site_id = model.site("left_foot_site").id
    right_site_id = model.site("right_foot_site").id

    with mujoco.viewer.launch_passive(model, sim) as viewer:
        i = 0
        while viewer.is_running():
            # Hip position
            hip_z = hip_height[i] / 1000.0
            hip_pos = np.array([0, 0, hip_z])

            # Foot positions relative to hip
            left_xy = foot_xy_rel_hip[i, 0] / 1000.0
            right_xy = foot_xy_rel_hip[i, 1] / 1000.0
            left_pos = hip_pos + np.array([left_xy[0], left_xy[1], -0.2 * foot_contacts[i, 0]])
            right_pos = hip_pos + np.array([right_xy[0], right_xy[1], -0.2 * foot_contacts[i, 1]])

            # Set site positions
            sim.site_xpos[hip_site_id] = hip_pos
            sim.site_xpos[left_site_id] = left_pos
            sim.site_xpos[right_site_id] = right_pos

            print(sim.site_xpos)

            mujoco.mj_forward(model, sim)
            viewer.sync()
            i = (i + 1) % n_frames
    print("Visualization finished.")


if __name__ == "__main__":
    ref_dir = Path(__file__).parent / "generated"
    if ref_dir.exists():
        for f in ref_dir.iterdir():
            if f.suffix == ".npz":
                print("Visualising:", f)
                visualise_reference_motion(f)
    else:
        print("No filepath found:", ref_dir)