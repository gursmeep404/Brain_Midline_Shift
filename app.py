from fastapi import FastAPI, File, UploadFile
import os
import numpy as np
import SimpleITK as sitk
import shutil
from src.segmentation import segment_volume_threshold
from src.preprocessing import preprocess_dicom
from src.midline_ideal import compute_ideal_midline_on_slice, get_visualization_slice_data
from src.midline_actual import estimate_actual_midline_mask_on_slice, get_actual_midline_data
from src.mls_calculator import calculate_midline_shift_mm
from src.visualisation import save_visualizations

app = FastAPI()

@app.post("/predict")
async def run_midline_shift_from_nifti(file: UploadFile = File(...)):
    # Step 1: Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 2: Load NIfTI
    volume = sitk.GetArrayFromImage(sitk.ReadImage(filepath))
    ref_img = sitk.ReadImage(filepath)

    # Step 3: Constants
    HU_MIN = -5
    HU_MAX = 75
    OUTPUT_DIR = "inference_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 4: Process pipeline (same as your main.py)
    ventricle_masks = segment_volume_threshold(volume)

    def get_ideal_slice_index(masks):
        max_area, best_index = 0, -1
        for i in range(masks.shape[0]):
            area = np.count_nonzero(masks[i])
            if area > max_area:
                max_area = area
                best_index = i
        return best_index

    ideal_slice_index = get_ideal_slice_index(ventricle_masks)

    slice_data, midline_mask_initial, _ = compute_ideal_midline_on_slice(
        ct_np=volume,
        ideal_slice_index=ideal_slice_index,
        output_dir=OUTPUT_DIR,
        hu_min=HU_MIN,
        hu_max=HU_MAX
    )

    ideal_img = sitk.GetImageFromArray(midline_mask_initial)
    ideal_img.CopyInformation(ref_img)
    ideal_path = os.path.join(OUTPUT_DIR, "ideal.nii.gz")
    sitk.WriteImage(ideal_img, ideal_path)

    actual_midline_mask = np.zeros_like(ventricle_masks, dtype=np.uint8)
    actual_midline_mask[ideal_slice_index] = estimate_actual_midline_mask_on_slice(ventricle_masks[ideal_slice_index])

    actual_img = sitk.GetImageFromArray(actual_midline_mask)
    actual_img.CopyInformation(ref_img)
    actual_path = os.path.join(OUTPUT_DIR, "actual.nii.gz")
    sitk.WriteImage(actual_img, actual_path)

    shifts_mm, avg_shift = calculate_midline_shift_mm(
        ideal_midline_path=ideal_path,
        actual_midline_path=actual_path,
        output_path=os.path.join(OUTPUT_DIR, "shifts.npz")
    )

    return {
        "ideal_slice_index": int(ideal_slice_index),
        "midline_shift_mm": float(avg_shift)
    }
