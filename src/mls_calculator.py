import SimpleITK as sitk
import numpy as np

def calculate_midline_shift_mm(ideal_midline_path, actual_midline_path, output_path=None):

    ideal_img = sitk.ReadImage(ideal_midline_path)
    actual_img = sitk.ReadImage(actual_midline_path)

    ideal_mask = sitk.GetArrayFromImage(ideal_img)  
    actual_mask = sitk.GetArrayFromImage(actual_img)

    spacing = ideal_img.GetSpacing()  
    x_spacing = spacing[0] 

    assert ideal_mask.shape == actual_mask.shape, "Masks must have same shape."

    max_area = -1
    best_index = -1

    for i in range(ideal_mask.shape[0]):
        count = np.count_nonzero(ideal_mask[i] == 1)
        if count > max_area:
            max_area = count
            best_index = i

    if best_index == -1:
        print("No valid slice found with ideal midline.")
        return [], 0

    ideal_xs = np.where(ideal_mask[best_index] > 0)[1]
    actual_xs = np.where(actual_mask[best_index] > 0)[1]


    if len(ideal_xs) == 0 or len(actual_xs) == 0:
        print("No valid midline pixels found in selected slice.")
        return [], 0

    mean_ideal_x = int(np.mean(ideal_xs))
    mean_actual_x = int(np.mean(actual_xs))

    pixel_shift = abs(mean_actual_x - mean_ideal_x)
    mm_shift = pixel_shift * x_spacing

    print(f"Midline shift (slice {best_index}): {mm_shift:.2f} mm")

    if output_path:
        np.savez(output_path,
                 slice_indices=[best_index],
                 shifts_mm=[mm_shift])

    return [mm_shift], mm_shift
