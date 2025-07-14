import numpy as np
import cv2

def estimate_actual_midline(mask):
  
    h, w = mask.shape
    midline_mask = np.zeros_like(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 2:
        return midline_mask  
    
    left_xs, right_xs = [], []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        if cx < w // 2:
            left_xs.append(cx)
        else:
            right_xs.append(cx)

    if not left_xs or not right_xs:
        return midline_mask 
    mean_left_x = int(np.mean(left_xs))
    mean_right_x = int(np.mean(right_xs))

   
    mid_x = int((mean_left_x + mean_right_x) / 2)

   
    midline_mask[:, mid_x-1:mid_x+2] = 255  

    return midline_mask
