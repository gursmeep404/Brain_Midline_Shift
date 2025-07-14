from src.midline_ideal import compute_midlines
from src.visualisation import save_visualizations
from src.segmentation import segment_volume_threshold
from src.preprocessing import preprocess_dicom
from src.midline_actual import estimate_actual_midline
import SimpleITK as sitk
import numpy as np
import os


USE_NIFTI = True  # Set to False to use DICOM
OUTPUT_DIR = "outputs/ventricles3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_CENTER = 35
WINDOW_WIDTH = 80
HU_MIN = WINDOW_CENTER - (WINDOW_WIDTH // 2)
HU_MAX = WINDOW_CENTER + (WINDOW_WIDTH // 2)


if USE_NIFTI:
    NIFTI_PATH = "registered_nifti_files/registered2.nii/registered2.nii"
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
    
    volume = preprocess_dicom(DICOM_PATH, hu_min=HU_MIN, hu_max=HU_MAX, resize_shape=(256, 256))

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

print("Calculating actual midline...")

midline_masks = np.zeros_like(ventricle_masks, dtype=np.uint8)

for i in range(ventricle_masks.shape[0]):
    vent_slice = ventricle_masks[i]
    midline = estimate_actual_midline(vent_slice)
    midline_masks[i] = midline


mid_img = sitk.GetImageFromArray(midline_masks)
mid_img.CopyInformation(ref_img)

sitk.WriteImage(mid_img, os.path.join(OUTPUT_DIR, "actual_midline_mask.nii.gz"))

print("Saved actual midline mask to NIfTI.")


print("Saving slice visualizations...")
save_visualizations(slices, OUTPUT_DIR, ventricle_masks=ventricle_masks)


print("Saving ventricle mask as NIfTI for 3D Slicer...")
vent_img = sitk.GetImageFromArray(ventricle_masks.astype(np.uint8))
vent_img.CopyInformation(ref_img)

sitk.WriteImage(vent_img, os.path.join(OUTPUT_DIR, "ventricle_mask.nii.gz"))

print("Done. Results saved in:", OUTPUT_DIR)
