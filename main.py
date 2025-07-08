import os
import numpy as np
from src.preprocessing import load_dicom_series, normalize_volume, resize_volume
from src.midline_ideal import estimate_ideal_midline
from src.segmentation import segment_ventricles
from src.midline_actual import estimate_actual_midline
from src.mls_calculator import calculate_mls
from src.visualisation import save_overlay

# Paths
dicom_folder = dicom_folder = r"data\Normal\105325641\20210122\brain"
output_folder = "data/outputs/patient_1"
os.makedirs(output_folder, exist_ok=True)

print("Loading DICOM series...")
volume = load_dicom_series(dicom_folder)
print(f"Original volume shape: {volume.shape}")

volume = normalize_volume(volume)
volume = resize_volume(volume)

mls_values = []

for i, slice_img in enumerate(volume):
    ideal_x = estimate_ideal_midline(slice_img)
    vent_mask = segment_ventricles(slice_img)
    actual_x = estimate_actual_midline(vent_mask)
    mls = calculate_mls(ideal_x, actual_x)
    mls_values.append(mls if mls is not None else 0)

    save_overlay(slice_img, ideal_x, actual_x, os.path.join(output_folder, f"slice_{i:02d}.png"))

mls_values = [v for v in mls_values if v is not None]

print("\n===== MIDLINE SHIFT RESULTS =====")
print(f"Max MLS (pixels): {np.max(mls_values):.2f}")
print(f"Mean MLS (pixels): {np.mean(mls_values):.2f}")
print(f"Total slices analyzed: {len(mls_values)}")