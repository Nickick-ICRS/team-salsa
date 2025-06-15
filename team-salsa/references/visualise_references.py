import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path


def load_tron1_model():
    urdf_path = Path(__file__).parent.parent / "SF_TRON1A/robot.urdf"
    assert urdf_path.exists
    return mujoco.MjModel.from_xml_path(str(urdf_path))


def solve_ik(model, sim, foot_targets, foot_contact, default_z=0.2, max_iters=100, tol=1e-4):
    """
    Simple iterative IK for both feet.
    foot_targets: (2, 3) array, target positions for left and right foot in world frame
    foot_contact: (2,) array, 1 if in contact, 0 if not
    """
    # Indices for left and right foot sites/geoms
    left_body_id = model.body("ankle_L_Link").id
    right_body_id = model.body("ankle_R_Link").id
    body_ids = [left_body_id, right_body_id]
    # Initial guess: keep current qpos
    qpos = sim.qpos.copy()
    for _ in range(max_iters):
        mujoco.mj_forward(model, sim)
        # Get current foot positions
        left_pos = sim.xpos[left_body_id]
        right_pos = sim.xpos[right_body_id]
        targets = [foot_targets[0], foot_targets[1]]
        foot_pos = [left_pos, right_pos]

        # Build error and jacobian
        err = []
        jac_list = []
        for i in range(2):
            err.append(targets[i] - foot_pos[i])
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, sim, jacp, None, body_ids[i])
            jac_list.append(jacp)
        if not err:
            break
        err = np.concatenate(err)
        jac = np.concatenate(jac_list, axis=0)  # (3*num_contacts, nv)

        # Damped least squares solution
        dls_lambda = 1e-4
        dq = np.linalg.lstsq(jac.T @ jac + dls_lambda * np.eye(model.nv), jac.T @ err, rcond=None)[0]
        qpos[7:model.nv] += dq[7:]
        sim.qpos[7:] = qpos[7:]
        mujoco.mj_forward(model, sim)
        if np.linalg.norm(err) < tol:
            break
        print(err)
    return sim.qpos.copy()


def visualise_reference(reference_path):
    data = np.load(reference_path)
    foot_contacts = data["foot_contacts"]
    foot_xy_rel_hip = data["foot_xy_rel_hip"]  # (n_frames, 2, 2)
    hip_height = data["hip_height"]            # (n_frames,)
    hip_orient_quat = data["hip_orient_quat"]  # (n_frames, 4)
    n_frames = foot_xy_rel_hip.shape[0]

    model = load_tron1_model()
    sim = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, sim) as viewer:
        i = 0
        while viewer.is_running():
            # Hip position and orientation
            hip_z = hip_height[i] / 100.0
            hip_quat = hip_orient_quat[i]
            # Compute world hip position
            hip_pos = np.array([0, 0, hip_z])
            # Compute foot targets in world frame
            left_xy = foot_xy_rel_hip[i, 0] / 100.0
            right_xy = foot_xy_rel_hip[i, 1] / 100.0
            left_target = hip_pos + np.array([left_xy[0], left_xy[1], -0.2])
            right_target = hip_pos + np.array([right_xy[0], right_xy[1], -0.2])
            foot_targets = np.stack([left_target, right_target])
            foot_contact = foot_contacts[i]
            # If not in contact, set z to default height
            for j in range(2):
                if not foot_contact[j]:
                    foot_targets[j][2] = 0.2
            # Solve IK for both feet (very simple, just for demo)
            sim.qpos[:] = solve_ik(model, sim, foot_targets, foot_contact)
            mujoco.mj_forward(model, sim)
            viewer.sync()

            i = (i+1) % n_frames
    print("Visualization finished.")


if __name__ == "__main__":
    ref_dir = Path(__file__).parent / "generated"
    filepaths = []
    if ref_dir.exists():
        print("Processing: ")
        for f in ref_dir.iterdir():
            filepaths.append(f)
            print(" - ", f)
            visualise_reference(f)
    else:
        print("No filepath found: ", ref_dir)
