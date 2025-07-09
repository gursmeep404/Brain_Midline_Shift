import cv2
import numpy as np

def save_overlay_rotated_midline(slice_gray, center_of_mass, angle, output_path, color=(0, 255, 0)):
    """
    Draw the tilted ideal midline based on rotation angle and center of mass.
    """
    h, w = slice_gray.shape
    img = cv2.cvtColor(slice_gray, cv2.COLOR_GRAY2BGR)

    theta = np.deg2rad(angle)
    dx = int(np.sin(theta) * h)
    dy = int(np.cos(theta) * h)

    x0 = int(center_of_mass[0] - dx // 2)
    y0 = int(center_of_mass[1] - dy // 2)
    x1 = int(center_of_mass[0] + dx // 2)
    y1 = int(center_of_mass[1] + dy // 2)

    cv2.line(img, (x0, y0), (x1, y1), color, 1)
    cv2.imwrite(output_path, img)
