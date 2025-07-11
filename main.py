# main.py
from ideal_midline import compute_midlines
from visualisation import save_visualizations

# === CONFIG ===
NIFTI_PATH = "registered_nifti_files/registered1.nii/registered1.nii"
OUTPUT_DIR = "output_midline_2"
WINDOW_CENTER = 35
WINDOW_WIDTH = 80
HU_MIN = WINDOW_CENTER - (WINDOW_WIDTH // 2)
HU_MAX = WINDOW_CENTER + (WINDOW_WIDTH // 2)

# === Run Midline Detection ===
print("[INFO] Running midline detection...")
slices, _ = compute_midlines(
    nifti_path=NIFTI_PATH,
    output_dir=OUTPUT_DIR,
    hu_min=HU_MIN,
    hu_max=HU_MAX
)

# === Save Visualizations ===
print("[INFO] Saving slice visualizations...")
save_visualizations(slices, OUTPUT_DIR)

# === Done ===
print(f"[DONE] Slices and masks saved to: {OUTPUT_DIR}")
print("Load CT with both midline masks in 3D Slicer to visualize.")
