import cv2
import numpy as np
from skimage.transform import rotate

def save_side_by_side_midline_visual(
    slice_gray, center_of_mass, angle, output_path, line_color=(0, 255, 0)
):
    """
    Visualize original and rotated CT slice side-by-side with midline shown.
    The rotated image has a vertical ideal midline drawn in the center.
    """
    h, w = slice_gray.shape

    # 1. Convert original to BGR
    original_bgr = cv2.cvtColor(slice_gray, cv2.COLOR_GRAY2BGR)

    # 2. Rotate the image using skimage (around center of mass)
    rotated = rotate(
        slice_gray, angle=angle, center=center_of_mass, preserve_range=True
    ).astype(np.uint8)

    # 3. Convert rotated to BGR
    rotated_bgr = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    # 4. Draw vertical midline on rotated image
    mid_x = rotated_bgr.shape[1] // 2
    cv2.line(rotated_bgr, (mid_x, 0), (mid_x, h - 1), line_color, 1)

    # 5. Stack images horizontally
    combined = np.hstack((original_bgr, rotated_bgr))

    # 6. Save the image
    cv2.imwrite(output_path, combined)
