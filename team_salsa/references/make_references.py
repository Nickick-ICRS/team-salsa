import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Assume YOLOv8 and pose model is available via ultralytics
from ultralytics import YOLO


def extract_salsa_references(video_path, output_path):
    # Load YOLO pose model
    model = YOLO('yolov8n-pose.pt')  # or your preferred pose model

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Storage for results
    foot_contacts = []
    foot_xy_rel_hip = []
    hip_orient_quat = []
    hip_xy_vel = []
    hip_height = []
    overlay_frames = []

    prev_hip_xy = None

    for _ in tqdm(range(n_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose estimation
        results = model(frame)
        keypoints = results[0].keypoints.xy.cpu().numpy()  # (num_people, num_keypoints, 2)
        if len(keypoints) == 0:
            # No person detected
            foot_contacts.append([0, 0])
            foot_xy_rel_hip.append([[0, 0], [0, 0]])
            hip_orient_quat.append([1, 0, 0, 0])
            hip_xy_vel.append([0, 0])
            hip_height.append(0)
            overlay_frames.append(frame)
            continue

        # Assume the first detected person is the dancer
        kp = keypoints[0]

        # Estimate pixel height (e.g., head to ankle)
        # COCO: 0 = nose, 15 = left ankle, 16 = right ankle
        head_y = kp[0][1]
        ankle_y = max(kp[15][1], kp[16][1])
        pixel_height = abs(ankle_y - head_y)
        if pixel_height < 1e-3:
            scale = 1.0  # fallback to avoid div by zero
        else:
            scale = 1.7 / pixel_height  # meters per pixel

        # Project 2D keypoints to 3D (x, y in image plane, z vertical from ground)
        keypoints_3d = []
        for x, y in kp:
            x3d = (x - frame.shape[1] / 2) * scale  # center x at 0
            y3d = (y - pixel_height) * -scale       # y=0 at bottom, up is positive
            z3d = 0                                 # Flat on ground, or use more info if available
            keypoints_3d.append([x3d, y3d, z3d])
        keypoints_3d = np.array(keypoints_3d)

        # Example: project hip center and ankles
        left_ankle_3d = keypoints_3d[15]
        right_ankle_3d = keypoints_3d[16]
        left_hip_3d = keypoints_3d[11]
        right_hip_3d = keypoints_3d[12]
        hip_center_3d = (left_hip_3d + right_hip_3d) / 2

        # Foot-floor contact: if y position of ankle is close to lowest y in frame (i.e., near ground)
        contact_thresh = 0.1
        left_contact = int(left_ankle_3d[2] > contact_thresh)
        right_contact = int(right_ankle_3d[2] > contact_thresh)
        foot_contacts.append([left_contact, right_contact])

        left_xyz = left_ankle_3d - hip_center_3d
        right_xyz = right_ankle_3d - hip_center_3d
        foot_xy_rel_hip.append([left_xyz.tolist()[:2], right_xyz.tolist()[:2]])

        # Hip orientation: estimate facing direction using shoulders or hips
        # Use vector from left hip to right hip as "x", and up as "z"
        hip_vec = right_hip_3d - left_hip_3d
        hip_vec = hip_vec / (np.linalg.norm(hip_vec) + 1e-8)
        angle = np.arctan2(hip_vec[1], hip_vec[0])
        # Convert to quaternion (z-rotation only)
        quat = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        hip_orient_quat.append(quat)

         # Hip velocity (in 3D, relative to orientation)
        if prev_hip_xy is not None:
            delta = hip_center_3d - prev_hip_xy
            # Rotate delta into hip orientation frame (about y)
            rot = np.array([
                [np.cos(-angle), 0, -np.sin(-angle)],
                [0, 1, 0],
                [np.sin(-angle), 0, np.cos(-angle)]
            ])
            rel_vel = rot @ delta
            rel_vel = rel_vel * fps
        else:
            rel_vel = np.zeros(3)
        hip_xy_vel.append(rel_vel.tolist())
        prev_hip_xy = hip_center_3d

        # Hip height relative to ground (distance from hip center to bottom of frame)
        hip_z = hip_center_3d[2]
        hip_height.append(float(hip_z))

        # Draw overlay and store
        overlay = draw_overlay(frame.copy(), kp, left_ankle_3d, right_ankle_3d, left_hip_3d, right_hip_3d, [left_contact, right_contact], left_xyz, right_xyz, hip_center_3d)
        overlay_frames.append(overlay)

    cap.release()

    # Save as numpy dict
    data = {
        "foot_contacts": np.array(foot_contacts),                # (n_frames, 2)
        "foot_xy_rel_hip": np.array(foot_xy_rel_hip),            # (n_frames, 2, 2)
        "hip_orient_quat": np.array(hip_orient_quat),            # (n_frames, 4)
        "hip_xy_vel": np.array(hip_xy_vel),                      # (n_frames, 2)
        "hip_height": np.array(hip_height),                      # (n_frames,)
        "fps": fps
    }
    np.savez_compressed(output_path, **data)
    print(f"Saved reference data to {output_path}")

    # Display overlay video in a loop
    print("Displaying overlay video. Press ESC to exit.")
    for idx in range(len(overlay_frames)):
        cv2.imshow("Salsa Reference Overlay", overlay_frames[idx])
        key = cv2.waitKey(int(1000 // fps))
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()


def draw_overlay(frame, kp, left_ankle, right_ankle, left_hip, right_hip, foot_contacts, left_xy, right_xy, hip_center):
    # Draw keypoints
    for x, y in kp:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    # Draw hips and ankles
    cv2.circle(frame, tuple(left_ankle.astype(int)), 6, (255, 0, 0), 2)
    cv2.circle(frame, tuple(right_ankle.astype(int)), 6, (0, 0, 255), 2)
    cv2.circle(frame, tuple(left_hip.astype(int)), 6, (255, 255, 0), 2)
    cv2.circle(frame, tuple(right_hip.astype(int)), 6, (0, 255, 255), 2)
    # Draw hip center
    cv2.circle(frame, tuple(hip_center.astype(int)), 6, (255, 0, 255), 2)
    # Draw lines from hip to feet
    cv2.line(frame, tuple(hip_center.astype(int)), tuple(left_ankle.astype(int)), (255, 0, 0), 2)
    cv2.line(frame, tuple(hip_center.astype(int)), tuple(right_ankle.astype(int)), (0, 0, 255), 2)
    # Draw contact state
    cv2.putText(frame, f"L: {'CONTACT' if foot_contacts[0] else 'AIR'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0) if foot_contacts[0] else (100,100,100), 2)
    cv2.putText(frame, f"R: {'CONTACT' if foot_contacts[1] else 'AIR'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if foot_contacts[1] else (100,100,100), 2)
    # Draw relative positions
    cv2.putText(frame, f"L rel: [{left_xy[0]:.1f}, {left_xy[1]:.1f}]", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
    cv2.putText(frame, f"R rel: [{right_xy[0]:.1f}, {right_xy[1]:.1f}]", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    return frame


if __name__ == "__main__":
    video_dir = Path(__file__).parent / "original_videos"
    outpur_dir = video_dir.parent / "generated"
    filepaths = []
    if video_dir.exists():
        print("Processing: ")
        for f in video_dir.iterdir():
            filepaths.append(f)
            o = outpur_dir / (f.stem + ".npz")
            print(" - ", f)
            extract_salsa_references(f, o)
            print("Saved to: ", o)
    else:
        print("No filepath found: ", video_dir)
