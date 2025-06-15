import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

def create_mujoco_model():
    # Simple 2D biped with two feet and a hip
    mjcf = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="hip" pos="0 0 1">
          <geom type="sphere" size="0.05" rgba="0.2 0.2 1 1"/>
          <body name="left_foot" pos="-0.1 0 -0.2">
            <geom type="sphere" size="0.03" rgba="1 0.2 0.2 1"/>
          </body>
          <body name="right_foot" pos="0.1 0 -0.2">
            <geom type="sphere" size="0.03" rgba="0.2 1 0.2 1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(mjcf)


def visualise_reference(reference_path):
    data = np.load(reference_path)
    foot_xy_rel_hip = data["foot_xy_rel_hip"]  # (n_frames, 2, 2)
    hip_height = data["hip_height"]            # (n_frames,)
    hip_orient_quat = data["hip_orient_quat"]  # (n_frames, 4)
    n_frames = foot_xy_rel_hip.shape[0]

    model = create_mujoco_model()
    sim = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, sim) as viewer:
        for i in range(n_frames):
            # Hip position
            hip_z = hip_height[i] / 100.0  # scale for visualization
            sim.qpos[0:3] = [0, 0, hip_z]

            # Hip orientation (only z-rotation used)
            # Mujoco expects [w, x, y, z]
            sim.qpos[3:7] = hip_orient_quat[i]

            # Left and right foot positions relative to hip
            left_xy = foot_xy_rel_hip[i, 0] / 100.0
            right_xy = foot_xy_rel_hip[i, 1] / 100.0

            # Set left foot
            model.body_pos[model.body('left_foot').id] = [left_xy[0], 0, -0.2]
            # Set right foot
            model.body_pos[model.body('right_foot').id] = [right_xy[0], 0, -0.2]

            mujoco.mj_step(model, sim)
            viewer.sync()
    print("Visualization finished.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="Path to reference .npz file")
    args = parser.parse_args()
    visualise_reference(args.reference)
