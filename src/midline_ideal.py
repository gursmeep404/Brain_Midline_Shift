import numpy as np
import cv2
from skimage.transform import rotate

def estimate_ideal_midline(slice_hu, threshold_hu=300, angle_range=(-15, 16), step=1):
   

    # 1. Threshold to get skull mask
    skull_mask = (slice_hu > threshold_hu).astype(np.uint8)

    # 2. Compute center of mass of skull
    y_coords, x_coords = np.nonzero(skull_mask)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return slice_hu.shape[1] // 2 

    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)
    center_of_mass = (int(x_center), int(y_center))

    # 3. Search for best rotation angle
    best_score = float('inf')
    best_angle = 0

    for angle in range(*angle_range, step):
        rotated = rotate(skull_mask, angle=angle, center=center_of_mass, preserve_range=True, order=0).astype(np.uint8)
        score = symmetry_score(rotated)
        if score < best_score:
            best_score = score
            best_angle = angle

    # 4. Use best angle to get ideal midline x position
    rotated = rotate(skull_mask, angle=best_angle, center=center_of_mass, preserve_range=True, order=0).astype(np.uint8)
    midline_x_in_rotated = rotated.shape[1] // 2

    # 5. Map this midline back to original image (reverse rotate)
    # We'll assume midline lies at x = midline_x_in_rotated in rotated image
    # So we back-rotate a vertical line and find where it intersects the original image

    # Create a binary image with vertical midline
    midline_img = np.zeros_like(rotated)
    midline_img[:, midline_x_in_rotated] = 1

    # Inverse rotate the midline line to original orientation
    unrotated_midline = rotate(midline_img, angle=-best_angle, center=center_of_mass, preserve_range=True, order=0).astype(np.uint8)

    # 6. Estimate the center X position of the back-rotated line
    midline_x_coords = np.where(unrotated_midline == 1)[1]
    if len(midline_x_coords) == 0:
        return slice_hu.shape[1] // 2
    ideal_midline_x = int(np.mean(midline_x_coords))

    return ideal_midline_x, best_angle, center_of_mass


# Compute symmetry score for a binary skull mask.

def symmetry_score(rotated_mask):

    h, w = rotated_mask.shape
    mid_x = w // 2
    total_diff = 0
    count = 0

    for y in range(h):
        row = rotated_mask[y]
        left = row[:mid_x][::-1]  
        right = row[mid_x:mid_x + len(left)]
        if len(left) == 0 or len(right) == 0:
            continue
        diff = np.abs(left - right)
        total_diff += np.sum(diff)
        count += len(diff)

    return total_diff / count if count > 0 else float('inf')
