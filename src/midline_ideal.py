import os
import numpy as np
import SimpleITK as sitk
import cv2
from scipy import ndimage

def compute_midlines(nifti_path=None, output_dir=None,
                     hu_min=-5, hu_max=75,
                     roi_height=50, roi_width=40,
                     canny_thresh_low=50, canny_thresh_high=150,
                     volume_override=None):

    if volume_override is not None:
        ct_np = volume_override
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0)
    else:
        import SimpleITK as sitk
        ct_img = sitk.ReadImage(nifti_path)
        ct_np = sitk.GetArrayFromImage(ct_img)
        spacing = ct_img.GetSpacing()
        origin = ct_img.GetOrigin()
        direction = ct_img.GetDirection()


    midline_mask_initial = np.zeros_like(ct_np, dtype=np.uint8)
    midline_mask_adjusted = np.zeros_like(ct_np, dtype=np.uint8)
    slice_data_list = []

    def detect_midline_x(roi, x_offset):
        edges = cv2.Canny(roi, canny_thresh_low, canny_thresh_high)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                                minLineLength=10, maxLineGap=5)
        x_coords = []
        if lines is not None:
            for line in lines:
                x1_, y1_, x2_, y2_ = line[0]
                if abs(x1_ - x2_) < 5:  # vertical-ish
                    x_mid = (x1_ + x2_) // 2
                    x_coords.append(x_mid + x_offset)
        return int(np.mean(x_coords)) if x_coords else None

    for i in range(ct_np.shape[0]):
        slice_hu = ct_np[i]
        bone_mask = ((slice_hu >= 300) & (slice_hu <= 2000)).astype(np.uint8)

        if np.sum(bone_mask) < 1000:
            continue

        _, initial_x = ndimage.center_of_mass(bone_mask)
        initial_x = int(initial_x)

        gray = np.clip(slice_hu, hu_min, hu_max)
        gray_norm = ((gray - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)

        h, w = gray_norm.shape
        x1 = max(0, initial_x - roi_width)
        x2 = min(w, initial_x + roi_width)

        top_roi = gray_norm[0:roi_height, x1:x2]
        bot_roi = gray_norm[-roi_height:, x1:x2]

        top_x = detect_midline_x(top_roi, x1)
        bot_x = detect_midline_x(bot_roi, x1)

        adjusted_x = int(np.mean([x for x in [top_x, bot_x] if x is not None])) if (top_x or bot_x) else initial_x

        midline_mask_initial[i, :, initial_x] = 1
        midline_mask_adjusted[i, :, adjusted_x] = 1

        slice_data_list.append({
            'index': i,
            'image': gray_norm,
            'initial_x': initial_x,
            'adjusted_x': adjusted_x
        })

    mask_init = sitk.GetImageFromArray(midline_mask_initial)
    mask_adj = sitk.GetImageFromArray(midline_mask_adjusted)
    for img in [mask_init, mask_adj]:
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        img.SetDirection(direction)

    sitk.WriteImage(mask_init, os.path.join(output_dir, "midline_center_of_mass.nii.gz"))
    sitk.WriteImage(mask_adj, os.path.join(output_dir, "midline_adjusted.nii.gz"))

    return slice_data_list, ct_np
