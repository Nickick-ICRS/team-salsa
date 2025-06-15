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
            continue

        # Assume the first detected person is the dancer
        kp = keypoints[0]

        # COCO format: 11 = left ankle, 12 = right ankle, 23 = left hip, 24 = right hip
        # For YOLOv8, check the keypoint order in the model you use!
        left_ankle = kp[15]
        right_ankle = kp[16]
        left_hip = kp[11]
        right_hip = kp[12]

        # Hip center
        hip_center = (left_hip + right_hip) / 2

        # Foot-floor contact: if y position of ankle is close to lowest y in frame (i.e., near ground)
        frame_height = frame.shape[0]
        contact_thresh = 0.98  # 98% of frame height (bottom)
        left_contact = int(left_ankle[1] > contact_thresh * frame_height)
        right_contact = int(right_ankle[1] > contact_thresh * frame_height)
        foot_contacts.append([left_contact, right_contact])

        # Foot xy position relative to hip center
        left_xy = left_ankle - hip_center
        right_xy = right_ankle - hip_center
        foot_xy_rel_hip.append([left_xy.tolist(), right_xy.tolist()])

        # Hip orientation: estimate facing direction using shoulders or hips
        # Use vector from left hip to right hip as "x", and up as "z"
        hip_vec = right_hip - left_hip
        hip_vec = hip_vec / (np.linalg.norm(hip_vec) + 1e-8)
        up_vec = np.array([0, -1])  # y axis up in image
        # 2D orientation: angle between hip_vec and x-axis
        angle = np.arctan2(hip_vec[1], hip_vec[0])
        # Convert to quaternion (z-rotation only)
        quat = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        hip_orient_quat.append(quat)

        # Hip xy velocity (relative to orientation)
        if prev_hip_xy is not None:
            delta = hip_center - prev_hip_xy
            # Rotate delta into hip orientation frame
            rot = np.array([[np.cos(-angle), -np.sin(-angle)],
                            [np.sin(-angle),  np.cos(-angle)]])
            rel_vel = rot @ delta
            rel_vel = rel_vel * fps  # pixels per second
        else:
            rel_vel = np.zeros(2)
        hip_xy_vel.append(rel_vel.tolist())
        prev_hip_xy = hip_center

        # Hip height relative to ground (distance from hip center to bottom of frame)
        hip_z = frame_height - hip_center[1]
        hip_height.append(float(hip_z))

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
