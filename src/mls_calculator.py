import SimpleITK as sitk
import numpy as np
import os

def calculate_midline_shift_mm(ideal_midline_path, actual_midline_path, output_path=None):
    # Load both midline masks
    ideal_img = sitk.ReadImage(ideal_midline_path)
    actual_img = sitk.ReadImage(actual_midline_path)

    ideal_mask = sitk.GetArrayFromImage(ideal_img)  
    actual_mask = sitk.GetArrayFromImage(actual_img)

    spacing = ideal_img.GetSpacing()  
    x_spacing = spacing[0]  

    assert ideal_mask.shape == actual_mask.shape, "Masks must have same shape."

    shifts_mm = []
    slices_considered = []

    for i in range(ideal_mask.shape[0]):
        ideal_xs = np.where(ideal_mask[i] == 1)[1]  
        actual_xs = np.where(actual_mask[i] == 255)[1] 

        if len(ideal_xs) == 0 or len(actual_xs) == 0:
            continue  # skip slices without data

        mean_ideal_x = int(np.mean(ideal_xs))
        # print(f"mean of ideal data {mean_ideal_x}")
        mean_actual_x = int(np.mean(actual_xs))
        # print(f"mean of actual data {mean_actual_x}")
        pixel_shift = abs(mean_actual_x - mean_ideal_x)
        mm_shift = pixel_shift * x_spacing

        shifts_mm.append(mm_shift)
        slices_considered.append(i)

    average_shift = np.mean(shifts_mm) if shifts_mm else 0

    print(f"Average midline shift: {average_shift:.2f} mm over {len(shifts_mm)} slices.")

    if output_path:
        np.savez(output_path, slice_indices=slices_considered, shifts_mm=shifts_mm)

    return shifts_mm, average_shift
