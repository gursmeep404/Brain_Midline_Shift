import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

# === CONFIGURATION ===
NIFTI_PATH = "registered_nifti_files/registered1.nii/registered1.nii"
OUTPUT_DIR = "output_midline_2"
WINDOW_CENTER = 35
WINDOW_WIDTH = 80
HU_MIN = WINDOW_CENTER - (WINDOW_WIDTH // 2)
HU_MAX = WINDOW_CENTER + (WINDOW_WIDTH // 2)
ROI_HEIGHT = 50
ROI_WIDTH = 40
CANNY_THRESH_LOW = 50
CANNY_THRESH_HIGH = 150

# Output filenames
MIDLINE_INITIAL_NAME = "midline_center_of_mass.nii.gz"
MIDLINE_ADJUSTED_NAME = "midline_adjusted.nii.gz"

# === Setup Directories ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "slices"), exist_ok=True)

# === Load NIfTI ===
ct_img = sitk.ReadImage(NIFTI_PATH)
ct_np = sitk.GetArrayFromImage(ct_img)
spacing = ct_img.GetSpacing()
origin = ct_img.GetOrigin()
direction = ct_img.GetDirection()

print(f"Loaded volume shape: {ct_np.shape}, HU range: {ct_np.min()} to {ct_np.max()}")

# === Prepare midline masks ===
midline_mask_initial = np.zeros_like(ct_np, dtype=np.uint8)
midline_mask_adjusted = np.zeros_like(ct_np, dtype=np.uint8)

# === Slice-wise Processing ===
for i in range(ct_np.shape[0]):
    slice_hu = ct_np[i]

    # Compute skull mask for initial midline
    bone_mask = ((slice_hu >= 300) & (slice_hu <= 2000)).astype(np.uint8)
    if np.sum(bone_mask) < 1000:
        print(f"Skipping slice {i}: too few bone pixels.")
        continue

    # === Initial midline ===
    _, initial_x = ndimage.center_of_mass(bone_mask)
    initial_x = int(initial_x)

    # === Adjusted midline using anatomical anchors ===
    gray = np.clip(slice_hu, HU_MIN, HU_MAX)
    gray_norm = ((gray - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)

    h, w = gray_norm.shape
    x1 = max(0, initial_x - ROI_WIDTH)
    x2 = min(w, initial_x + ROI_WIDTH)

    top_roi = gray_norm[0:ROI_HEIGHT, x1:x2]
    bot_roi = gray_norm[-ROI_HEIGHT:, x1:x2]

    def detect_midline_x(roi, x_offset):
        edges = cv2.Canny(roi, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                                minLineLength=10, maxLineGap=5)
        x_coords = []
        if lines is not None:
            for line in lines:
                x1_, y1_, x2_, y2_ = line[0]
                if abs(x1_ - x2_) < 5:  # near vertical
                    x_mid = (x1_ + x2_) // 2
                    x_coords.append(x_mid + x_offset)
        return int(np.mean(x_coords)) if x_coords else initial_x

    top_x = detect_midline_x(top_roi, x1)
    bot_x = detect_midline_x(bot_roi, x1)
    adjusted_x = int(np.mean([top_x, bot_x]))

    # === Save to 3D mask ===
    midline_mask_initial[i, :, initial_x] = 1
    midline_mask_adjusted[i, :, adjusted_x] = 1

    # === Visualization ===
    slice_bgr = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR)
    cv2.line(slice_bgr, (initial_x, 0), (initial_x, h-1), (0, 255, 0), 2)  # green = initial
    cv2.line(slice_bgr, (adjusted_x, 0), (adjusted_x, h-1), (0, 0, 255), 2)  # red = adjusted

    out_path = os.path.join(OUTPUT_DIR, "slices", f"slice_{i:03d}.png")
    cv2.imwrite(out_path, slice_bgr)

    print(f"Slice {i:03d} | Initial: {initial_x}, Adjusted: {adjusted_x}")

# === Save 3D Masks ===
img_initial = sitk.GetImageFromArray(midline_mask_initial)
img_initial.SetSpacing(spacing)
img_initial.SetOrigin(origin)
img_initial.SetDirection(direction)
sitk.WriteImage(img_initial, os.path.join(OUTPUT_DIR, MIDLINE_INITIAL_NAME))

img_adjusted = sitk.GetImageFromArray(midline_mask_adjusted)
img_adjusted.SetSpacing(spacing)
img_adjusted.SetOrigin(origin)
img_adjusted.SetDirection(direction)
sitk.WriteImage(img_adjusted, os.path.join(OUTPUT_DIR, MIDLINE_ADJUSTED_NAME))

print(f"\n Slices saved in: {OUTPUT_DIR}/slices")
print(f"Initial midline: {MIDLINE_INITIAL_NAME}")
print(f"Adjusted midline: {MIDLINE_ADJUSTED_NAME}")
print("Load these masks in 3D Slicer with the CT to visualize both midlines.")
