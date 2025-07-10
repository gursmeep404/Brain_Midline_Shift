# visualisation.py
import cv2
import numpy as np

def save_midline_visual(
    slice_gray,
    midline_x,
    output_path,
    midline_color=(0, 255, 0)
):
    h, _ = slice_gray.shape
    img = cv2.cvtColor(slice_gray, cv2.COLOR_GRAY2BGR)
    cv2.line(img, (midline_x, 0), (midline_x, h - 1), midline_color, 2)
    cv2.imwrite(output_path, img)
