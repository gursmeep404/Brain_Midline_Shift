import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

# === CONFIGURATION ===
NIFTI_PATH = "registered_nifti_files/registered1.nii/registered1.nii"  
OUTPUT_DIR = "output_midline"
MIDLINE_MASK_NAME = "midline_mask.nii.gz"
WINDOW_CENTER = 35
WINDOW_WIDTH = 80
HU_MIN = WINDOW_CENTER - (WINDOW_WIDTH // 2)  
HU_MAX = WINDOW_CENTER + (WINDOW_WIDTH // 2) 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "slices"), exist_ok=True)

# === Load NIfTI File ===
ct_img = sitk.ReadImage(NIFTI_PATH)
ct_np = sitk.GetArrayFromImage(ct_img)  # [Z, Y, X]
spacing = ct_img.GetSpacing()
origin = ct_img.GetOrigin()
direction = ct_img.GetDirection()

print(f"Loaded volume shape: {ct_np.shape}, HU range: {ct_np.min()} to {ct_np.max()}")

# === Prepare blank midline mask ===
midline_mask = np.zeros_like(ct_np, dtype=np.uint8)

# === Process Each Slice ===
for i in range(ct_np.shape[0]):
    slice_hu = ct_np[i]

    # Compute bone/skull mask for center of mass
    bone_mask = ((slice_hu >= 300) & (slice_hu <= 2000)).astype(np.uint8)

    if np.sum(bone_mask) < 1000:
        print(f"Skipping slice {i}: too few bone pixels.")
        continue

    center_y, center_x = ndimage.center_of_mass(bone_mask)
    mid_x = int(center_x)

    # Store line in midline_mask volume (vertical line at mid_x)
    midline_mask[i, :, mid_x] = 1  # Entire column in that slice

    # For visualization: window HU and draw line
    slice_windowed = np.clip(slice_hu, HU_MIN, HU_MAX)
    slice_norm = ((slice_windowed - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    slice_bgr = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2BGR)
    cv2.line(slice_bgr, (mid_x, 0), (mid_x, slice_bgr.shape[0] - 1), (0, 0, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, "slices", f"slice_{i:03d}.png")
    cv2.imwrite(out_path, slice_bgr)

# === Save midline mask as NIfTI ===
midline_mask_img = sitk.GetImageFromArray(midline_mask)
midline_mask_img.SetSpacing(spacing)
midline_mask_img.SetOrigin(origin)
midline_mask_img.SetDirection(direction)

sitk.WriteImage(midline_mask_img, os.path.join(OUTPUT_DIR, MIDLINE_MASK_NAME))

print(f"\n Midline visualization saved to: {OUTPUT_DIR}/slices")
print(f"3D Midline mask saved as: {os.path.join(OUTPUT_DIR, MIDLINE_MASK_NAME)}")
print("Load both CT and midline_mask.nii.gz in 3D Slicer to visualize.")
