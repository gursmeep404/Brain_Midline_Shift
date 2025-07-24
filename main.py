import os
import numpy as np
import SimpleITK as sitk

from src.segmentation import segment_volume_threshold
from src.preprocessing import preprocess_dicom
from src.midline_ideal import compute_ideal_midline_on_slice
from src.midline_ideal import get_visualization_slice_data
from src.midline_actual import estimate_actual_midline_mask_on_slice, get_actual_midline_data
from src.mls_calculator import calculate_midline_shift_mm
from src.visualisation import save_visualizations

USE_NIFTI = True  # Set to False to use DICOM
OUTPUT_DIR = "testing/test_sample_mls_again_6_idealslice_again"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_CENTER = 35
WINDOW_WIDTH = 80
HU_MIN = WINDOW_CENTER - (WINDOW_WIDTH // 2)
HU_MAX = WINDOW_CENTER + (WINDOW_WIDTH // 2)


if USE_NIFTI:
    NIFTI_PATH = "registered_nifti_files/registered_mls_6.nii/registered_mls_6.nii"
    print("Running midline detection from NIFTI...")
    volume = sitk.GetArrayFromImage(sitk.ReadImage(NIFTI_PATH))
    ref_img = sitk.ReadImage(NIFTI_PATH)
else:
    DICOM_PATH = "dicom_files"
    print("Preprocessing DICOM series...")
    volume = preprocess_dicom(DICOM_PATH, hu_min=HU_MIN, hu_max=HU_MAX)
    ref_img = sitk.GetImageFromArray(volume)
    ref_img.SetSpacing((1.0, 1.0, 1.0))
    ref_img.SetOrigin((0.0, 0.0, 0.0))
    ref_img.SetDirection((1.0, 0.0, 0.0))

print("Segmenting ventricles using thresholding...")
ventricle_masks = segment_volume_threshold(volume)

print("Identifying ideal slice with max ventricle area...")
def get_ideal_slice_index(ventricle_masks):
    max_area, best_index = 0, -1
    for i in range(ventricle_masks.shape[0]):
        area = np.count_nonzero(ventricle_masks[i])
        if area > max_area:
            max_area = area
            best_index = i
    return best_index

ideal_slice_index = get_ideal_slice_index(ventricle_masks)
print(f"Ideal slice selected: {ideal_slice_index}")

print("Computing ideal midline for ideal slice...")
slice_data, midline_mask_initial, midline_mask_adjusted = compute_ideal_midline_on_slice(
    ct_np=volume,
    ideal_slice_index=ideal_slice_index,
    output_dir=OUTPUT_DIR,
    hu_min=HU_MIN,
    hu_max=HU_MAX
)

ideal_midline_mask = midline_mask_initial

ideal_img = sitk.GetImageFromArray(ideal_midline_mask)
ideal_img.CopyInformation(ref_img)
sitk.WriteImage(ideal_img, os.path.join(OUTPUT_DIR, "midline_center_of_mass.nii.gz"))

print("Estimating actual midline for ideal slice...")
actual_midline_mask = np.zeros_like(ventricle_masks, dtype=np.uint8)
actual_midline_mask[ideal_slice_index] = estimate_actual_midline_mask_on_slice(ventricle_masks[ideal_slice_index])

actual_img = sitk.GetImageFromArray(actual_midline_mask)
actual_img.CopyInformation(ref_img)
sitk.WriteImage(actual_img, os.path.join(OUTPUT_DIR, "actual_midline_mask.nii.gz"))

print("Calculating midline shift...")
shifts_mm, avg_shift = calculate_midline_shift_mm(
    ideal_midline_path=os.path.join(OUTPUT_DIR, "midline_center_of_mass.nii.gz"),
    actual_midline_path=os.path.join(OUTPUT_DIR, "actual_midline_mask.nii.gz"),
    output_path=os.path.join(OUTPUT_DIR, "midline_shifts_mm.npz")
)


print("Saving slice visualizations...")
slice_data = get_visualization_slice_data(volume, ideal_slice_index, HU_MIN, HU_MAX)
actual_midline_data = get_actual_midline_data(volume,slice_index=ideal_slice_index)

save_visualizations(
    [slice_data],
    OUTPUT_DIR,
    ventricle_masks=[ventricle_masks[ideal_slice_index]],
    actual_midline_data=[actual_midline_data]  # wrap in a list!
)

print("Saving ventricle mask as NIfTI...")
vent_img = sitk.GetImageFromArray(ventricle_masks.astype(np.uint8))
vent_img.CopyInformation(ref_img)
sitk.WriteImage(vent_img, os.path.join(OUTPUT_DIR, "ventricle_mask.nii.gz"))

