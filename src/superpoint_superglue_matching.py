"""
SuperPoint + SuperGlue keypoint matching between object images and scene images.

For every (object, scene) pair this script:
  1. detects keypoints and descriptors with SuperPoint,
  2. matches them with SuperGlue (indoor weights),
  3. saves a side-by-side visualization of the matches,
  4. saves runtime / keypoint / match statistics to a .txt file,
  5. saves the matched scene keypoint coordinates to a .csv file
     (used later for evaluation against the ground-truth masks).

Expected layout (relative to the repository root):
  data/objects/   object crops (.png)
  data/scenes/    scene images (.png)
  outputs/        created automatically
  src/SuperGluePretrainedNetwork/   clone of magicleap/SuperGluePretrainedNetwork
"""

import sys
import os
import cv2
import torch
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend, only used for saving images
import matplotlib.pyplot as plt

# Make the SuperGlue repository importable (git clone name or GitHub zip name)
script_dir = os.path.dirname(os.path.abspath(__file__))
for folder in ('SuperGluePretrainedNetwork', 'SuperGluePretrainedNetwork-master'):
    sys.path.append(os.path.join(script_dir, folder))

from models.superpoint import SuperPoint
from models.superglue import SuperGlue


total_time_start = time.time()

# Use the GPU if one is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SuperPoint initialization
superpoint = SuperPoint({}).to(device)
superpoint.eval()
# SuperGlue initialization with indoor weights
superglue = SuperGlue({
    'weights': 'indoor',
    'match_threshold': 0.01
}).to(device)
superglue.eval()


# Folders
base_dir = os.path.dirname(script_dir)
objects_dir = os.path.join(base_dir, 'data', 'objects')
scenes_dir = os.path.join(base_dir, 'data', 'scenes')
outputs_dir = os.path.join(base_dir, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

# Load file names (png)
object_files = sorted([f for f in os.listdir(objects_dir) if f.endswith('.png')])
scene_files = sorted([f for f in os.listdir(scenes_dir) if f.endswith('.png')])


for obj_file in object_files:
    start_time0 = time.time()
    obj_path = os.path.join(objects_dir, obj_file)
    # Load the object image
    object_img = cv2.imread(obj_path)
    # Convert to grayscale
    object_gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
    # Convert to a normalized tensor
    object_tensor = torch.tensor(object_gray, dtype=torch.float32) / 255.0
    # Reshape to [1, 1, H, W]
    object_tensor = object_tensor.unsqueeze(0).unsqueeze(0).to(device)
    end_time0 = time.time()

    # Detect keypoints and descriptors on the object
    with torch.no_grad():
        object_output = superpoint({'image': object_tensor})

    for scene_file in scene_files:
        start_time1 = time.time()
        scene_path = os.path.join(scenes_dir, scene_file)
        scene_img = cv2.imread(scene_path)
        scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
        scene_tensor = torch.tensor(scene_gray, dtype=torch.float32) / 255.0
        scene_tensor = scene_tensor.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            scene_output = superpoint({'image': scene_tensor})

        # Match the two keypoint sets with SuperGlue
        with torch.no_grad():
            kpts0 = object_output['keypoints'][0]
            kpts1 = scene_output['keypoints'][0]
            desc0 = object_output['descriptors'][0]
            desc1 = scene_output['descriptors'][0]
            scores0 = object_output['scores'][0]
            scores1 = scene_output['scores'][0]

            if isinstance(kpts0, np.ndarray):
                kpts0 = torch.from_numpy(kpts0)
            if isinstance(kpts1, np.ndarray):
                kpts1 = torch.from_numpy(kpts1)

            kpts0 = kpts0.unsqueeze(0).to(device)      # [1, N, 2]
            kpts1 = kpts1.unsqueeze(0).to(device)
            desc0 = desc0.unsqueeze(0).to(device)      # [1, 256, N]
            desc1 = desc1.unsqueeze(0).to(device)
            scores0 = scores0.unsqueeze(0).to(device)  # [1, N]
            scores1 = scores1.unsqueeze(0).to(device)

            match_output = superglue({
                'keypoints0': kpts0,
                'keypoints1': kpts1,
                'descriptors0': desc0,
                'descriptors1': desc1,
                'scores0': scores0,
                'scores1': scores1,
                'image0': object_tensor,
                'image1': scene_tensor,
            })

        matches = match_output['matches0'][0].cpu().numpy()  # Index of the matched point, or -1 if unmatched
        num_matches = np.sum(matches > -1)

        # Convert to RGB for saving
        object_rgb = cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB)
        scene_rgb = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)

        # Place the object and scene images side by side
        h1, w1, _ = object_rgb.shape
        h2, w2, _ = scene_rgb.shape
        height = max(h1, h2)
        combined_img = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
        combined_img[:h1, :w1] = object_rgb
        combined_img[:h2, w1:] = scene_rgb

        # Coordinates of the matched keypoints
        kpts0_np = kpts0[0].cpu().numpy()  # Remove batch dimension -> [N, 2]
        kpts1_np = kpts1[0].cpu().numpy()

        matched_kpts0 = kpts0_np[matches > -1]
        matched_kpts1 = kpts1_np[matches[matches > -1]]

        # Draw each match as two dots connected by a line
        for pt0, pt1 in zip(matched_kpts0, matched_kpts1):
            pt0 = tuple(map(int, pt0))
            pt1 = tuple(map(int, pt1))
            pt1_shifted = (int(pt1[0] + w1), int(pt1[1]))  # Shift right into the scene half
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.circle(combined_img, pt0, 3, color, -1)
            cv2.circle(combined_img, pt1_shifted, 3, color, -1)
            cv2.line(combined_img, pt0, pt1_shifted, color, 1)

        # Output names: <object>_<scene>.png / .txt / .csv
        obj_num = os.path.splitext(obj_file)[0]
        scene_num = os.path.splitext(scene_file)[0]
        output_img_path = os.path.join(outputs_dir, f"{obj_num}_{scene_num}.png")
        output_txt_path = os.path.join(outputs_dir, f"{obj_num}_{scene_num}.txt")
        output_csv_path = os.path.join(outputs_dir, f"{obj_num}_{scene_num}.csv")

        plt.imsave(output_img_path, combined_img)

        # Runtime for this object-scene pair
        end_time1 = time.time()
        elapsed_time = end_time1 - start_time1 + (end_time0 - start_time0)

        # Save statistics
        with open(output_txt_path, 'w') as f:
            f.write(f"Runtime (seconds): {elapsed_time:.4f}\n")
            f.write(f"Object keypoints: {len(object_output['keypoints'][0])}\n")
            f.write(f"Scene keypoints: {len(scene_output['keypoints'][0])}\n")
            f.write(f"Matches found: {num_matches}\n")

        # Save the matched scene keypoints (x, y) for evaluation against ground-truth masks
        matched_scene_coords = kpts1_np[matches[matches > -1]]
        np.savetxt(output_csv_path, matched_scene_coords, delimiter=',', fmt='%.2f', header='x,y', comments='')

# Total time for all object-scene combinations (6 objects x 20 scenes = 120)
total_time_end = time.time()
print("Total elapsed time (s):", total_time_end - total_time_start)
