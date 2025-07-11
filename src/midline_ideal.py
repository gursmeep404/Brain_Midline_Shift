import numpy as np
import cv2
import SimpleITK as sitk


# def register_ct_to_atlas(dicom_folder, atlas_path):
#     reader = sitk.ImageSeriesReader()
#     series_IDs = reader.GetGDCMSeriesIDs(dicom_folder)
#     if not series_IDs:
#         raise ValueError(f"No DICOM series found in {dicom_folder}")
#     dicom_filenames = reader.GetGDCMSeriesFileNames(dicom_folder, series_IDs[0])
#     reader.SetFileNames(dicom_filenames)
#     ct_image = reader.Execute()

#     atlas_image = sitk.ReadImage(atlas_path)
#     atlas_image = sitk.Cast(atlas_image, sitk.sitkFloat32)
#     ct_image = sitk.Cast(ct_image, sitk.sitkFloat32)

#     transform = sitk.CenteredTransformInitializer(
#         atlas_image,
#         ct_image,
#         sitk.Euler3DTransform(),
#         sitk.CenteredTransformInitializerFilter.GEOMETRY
#     )

#     registration_method = sitk.ImageRegistrationMethod()
#     registration_method.SetMetricAsMattesMutualInformation(50)
#     registration_method.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, 100)
#     registration_method.SetInterpolator(sitk.sitkLinear)
#     registration_method.SetInitialTransform(transform, inPlace=False)
#     final_transform = registration_method.Execute(atlas_image, ct_image)

#     resampled = sitk.Resample(
#         ct_image,
#         atlas_image,
#         final_transform,
#         sitk.sitkLinear,
#         0.0,
#         ct_image.GetPixelID()
#     )
#     return sitk.GetArrayFromImage(resampled)


def segment_skull(slice_hu, hu_threshold=(100, 2000)):
    mask = ((slice_hu > hu_threshold[0]) & (slice_hu < hu_threshold[1])).astype(np.uint8)
    mask = cv2.medianBlur(mask, 3)
    return mask


def estimate_midline_from_com(mask):
    y, x = np.nonzero(mask)
    if len(x) == 0:
        print("Warning: empty mask.")
        return mask.shape[1] // 2  # fallback
    return int(np.mean(x))


# def refine_midline_with_anatomy(slice_hu, ideal_x, band=20, return_debug=False):
#     h, w = slice_hu.shape
#     bone_mask = (slice_hu > 100).astype(np.uint8)
#     y_coords, _ = np.nonzero(bone_mask)

#     if len(y_coords) > 0:
#         top = np.min(y_coords)
#         bottom = np.max(y_coords)
#         upper_y_range = (max(0, top + 5), min(h, top + 60))
#         lower_y_range = (max(0, bottom - 60), min(h, bottom - 5))
#     else:
#         upper_y_range = (20, 80)
#         lower_y_range = (h - 100, h)

#     x0 = max(0, ideal_x - band)
#     x1 = min(w, ideal_x + band)
#     upper_roi = slice_hu[upper_y_range[0]:upper_y_range[1], x0:x1]
#     lower_roi = slice_hu[lower_y_range[0]:lower_y_range[1], x0:x1]

#     def normalize_roi(roi, min_hu=-100, max_hu=300):
#         roi = np.clip(roi, min_hu, max_hu)
#         return ((roi - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)

#     upper_roi_norm = normalize_roi(upper_roi)
#     lower_roi_norm = normalize_roi(lower_roi)
#     edges_upper = cv2.Canny(upper_roi_norm, 50, 150)
#     edges_lower = cv2.Canny(lower_roi_norm, 50, 150)

#     def get_vertical_lines(edge_img, roi_origin_x, angle_thresh=np.deg2rad(10)):
#         lines = cv2.HoughLines(edge_img, rho=1, theta=np.pi / 180, threshold=20)
#         x_positions = []
#         if lines is not None:
#             for line in lines:
#                 rho, theta = line[0]
#                 if abs(theta - np.pi/2) < angle_thresh:
#                     x = int(rho / np.cos(theta)) + roi_origin_x
#                     if 0 <= x < w:
#                         x_positions.append(x)
#         return x_positions

#     upper_lines = get_vertical_lines(edges_upper, x0, np.deg2rad(10))
#     lower_lines = get_vertical_lines(edges_lower, x0, np.deg2rad(15))

#     def detect_anterior_skull_point(bone_mask):
#         for y in range(0, h // 4):
#             row = bone_mask[y]
#             x_indices = np.where(row > 0)[0]
#             if len(x_indices) > 0:
#                 return x_indices[len(x_indices) // 2]
#         return None

#     anterior_x = detect_anterior_skull_point(bone_mask)

#     anchors = [ideal_x]
#     weights = [1.0]

#     if anterior_x is not None:
#         anchors.append(anterior_x)
#         weights.append(1.5)

#     for x in upper_lines:
#         anchors.append(x)
#         weights.append(2.0)

#     for x in lower_lines:
#         anchors.append(x)
#         weights.append(1.0)

#     refined_x = int(np.average(anchors, weights=weights)) if len(anchors) > 0 else ideal_x

#     if return_debug:
#         debug = {
#             'edges_upper': edges_upper,
#             'edges_lower': edges_lower,
#             'upper_y_range': upper_y_range,
#             'lower_y_range': lower_y_range,
#             'band': band,
#             'anterior_x': anterior_x,
#             'upper_lines': upper_lines,
#             'lower_lines': lower_lines,
#             'anchors': anchors,
#             'weights': weights
#         }
#         return refined_x, debug
#     else:
#         return refined_x