from src.midline_ideal import compute_midlines
from src.visualisation import save_visualizations
from src.segmentation import segment_volume_threshold
from src.preprocessing import preprocess_dicom
from src.midline_actual import estimate_actual_midline_mask
from src.mls_calculator import calculate_midline_shift_mm
from src.midline_actual import get_actual_midline_data
import SimpleITK as sitk
import numpy as np
import os


USE_NIFTI = True  # Set to False to use DICOM
OUTPUT_DIR = "testing/test_sample_normal_again_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_CENTER = 35
WINDOW_WIDTH = 80
HU_MIN = WINDOW_CENTER - (WINDOW_WIDTH // 2)
HU_MAX = WINDOW_CENTER + (WINDOW_WIDTH // 2)


if USE_NIFTI:
    NIFTI_PATH = "registered_nifti_files/registered_normal_1.nii/registered_normal_1.nii"
    print("Running midline detection from NIFTI...")
    
    slices, volume = compute_midlines(
        nifti_path=NIFTI_PATH,
        output_dir=OUTPUT_DIR,
        hu_min=HU_MIN,
        hu_max=HU_MAX
    )
    
    
    ref_img = sitk.ReadImage(NIFTI_PATH)
else:
    DICOM_PATH = "dicom_files"
    print("Preprocessing DICOM series...")
    
    volume = preprocess_dicom(DICOM_PATH, hu_min=HU_MIN, hu_max=HU_MAX)

    print("Running midline detection on preprocessed volume...")
    slices, volume = compute_midlines(
        nifti_path=None,
        output_dir=OUTPUT_DIR,
        hu_min=HU_MIN,
        hu_max=HU_MAX,
        volume_override=volume
    )

  
    ref_img = sitk.GetImageFromArray(volume)
    ref_img.SetSpacing((1.0, 1.0, 1.0))
    ref_img.SetOrigin((0.0, 0.0, 0.0))
    ref_img.SetDirection((1.0, 0.0, 0.0))



print("Segmenting ventricles using thresholding...")
ventricle_masks = segment_volume_threshold(volume)


TEMPLATE_DIR = "templates" 

print("Estimating actual midline ...")

midline_masks = estimate_actual_midline_mask(ventricle_masks, TEMPLATE_DIR)

# Save to NIfTI
midline_img = sitk.GetImageFromArray(midline_masks.astype(np.uint8))
midline_img.CopyInformation(ref_img)
sitk.WriteImage(midline_img, os.path.join(OUTPUT_DIR, "actual_midline_mask.nii.gz"))
print("Saved actual midline mask to NIfTI.")


ideal_midline_path = os.path.join(OUTPUT_DIR, "midline_center_of_mass.nii.gz")
actual_midline_path = os.path.join(OUTPUT_DIR, "actual_midline_mask.nii.gz")
output_shift_path = os.path.join(OUTPUT_DIR, "midline_shifts_mm.npz")

shifts_mm, avg_shift = calculate_midline_shift_mm(
    ideal_midline_path=ideal_midline_path,
    actual_midline_path=actual_midline_path,
    output_path=output_shift_path
)



print("Saving slice visualizations...")
actual_midline_data = get_actual_midline_data()


save_visualizations(
    slices,
    OUTPUT_DIR,
    ventricle_masks=ventricle_masks,
    actual_midline_data=actual_midline_data
)


print("Saving ventricle mask as NIfti...")
vent_img = sitk.GetImageFromArray(ventricle_masks.astype(np.uint8))
vent_img.CopyInformation(ref_img)

sitk.WriteImage(vent_img, os.path.join(OUTPUT_DIR, "ventricle_mask.nii.gz"))

print("Done. Results saved in:", OUTPUT_DIR)
