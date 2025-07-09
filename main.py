import os
import numpy as np
from src.preprocessing import load_dicom_series, normalize_volume  # removed resize_volume
from src.midline_ideal import estimate_ideal_midline
from src.segmentation import segment_ventricles
from src.midline_actual import estimate_actual_midline
from src.mls_calculator import calculate_mls
from src.visualisation import save_side_by_side_midline_visual


dicom_folder = r"data\Normal\105325641\20210122\brain"
output_folder = "data/outputs/patient_1"
os.makedirs(output_folder, exist_ok=True)

print("Loading DICOM series...")
hu_volume = load_dicom_series(dicom_folder)  
print(f"Original volume shape: {hu_volume.shape}")

# Normalized volume only for visualization
vis_volume = normalize_volume(hu_volume)

mls_values = []

for i in range(len(hu_volume)):
    slice_hu = hu_volume[i]         
    slice_vis = vis_volume[i]       

    # --- Midline estimation ---
    ideal_x, angle, center_of_mass = estimate_ideal_midline(slice_hu)


    # --- Ventricle segmentation ---
    # vent_mask = segment_ventricles(slice_hu)    

    # --- Actual midline from ventricle mask ---
    # actual_x = estimate_actual_midline(vent_mask)

    # --- Midline Shift (in pixels) ---
    # mls = calculate_mls(ideal_x, actual_x)
    # mls_values.append(mls if mls is not None else 0)

    # --- Save overlay for visualization ---
    # from src.visualisation import save_overlay_symmetry_only

   
    save_side_by_side_midline_visual(
        slice_vis,
        center_of_mass=center_of_mass,
        angle=angle,
        output_path=os.path.join(output_folder, f"symmetry_pair_{i:02d}.png")
    )




# --- Report ---
mls_values = [v for v in mls_values if v is not None]

print("\n===== MIDLINE SHIFT RESULTS =====")
# print(f"Max MLS (pixels): {np.max(mls_values):.2f}")
# print(f"Mean MLS (pixels): {np.mean(mls_values):.2f}")
# print(f"Total slices analyzed: {len(mls_values)}")
print("done")