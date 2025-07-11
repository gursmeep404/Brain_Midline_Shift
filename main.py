import os
import numpy as np
from src.preprocessing import normalize_volume
from src.midline_ideal import register_ct_to_atlas, segment_skull, estimate_midline_from_com
from src.visualisation import save_midline_visual

dicom_folder = r"data\MidlineshiftData\105268689\20210122\brain"
atlas_path = r"data\atlas\MNI152_T1_2mm.nii.gz"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

print("Registering CT volume to brain atlas...")
registered_volume = register_ct_to_atlas(dicom_folder, atlas_path)

print("Normalizing for visualization...")
def normalize(img, min_hu=-1000, max_hu=1000):
    img = np.clip(img, min_hu, max_hu)
    return ((img - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
vis_volume = np.array([normalize(s) for s in registered_volume])

print("Processing slices...")
for i in range(len(registered_volume)):
    slice_hu = registered_volume[i]
    slice_vis = vis_volume[i]

    skull_mask = segment_skull(slice_hu)
    midline_x = estimate_midline_from_com(skull_mask)

    save_midline_visual(
        slice_vis,
        midline_x=midline_x,
        output_path=os.path.join(output_folder, f"midline_{i:03d}.png")
    )

print("Done.")
