import matplotlib.pyplot as plt
import os
import cv2

def save_overlay(slice_img, ideal_x, actual_x, save_path):
    img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
    h = slice_img.shape[0]
    if ideal_x is not None:
        cv2.line(img_rgb, (ideal_x, 0), (ideal_x, h), (0, 255, 0), 1)
    if actual_x is not None:
        cv2.line(img_rgb, (actual_x, 0), (actual_x, h), (0, 0, 255), 1)
    cv2.imwrite(save_path, img_rgb)