import numpy as np
import cv2

# Determines if a slice contains valid left and right ventricular regions.

def is_valid_ventricle_slice(mask, min_area=100):
  
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    left_area, right_area = 0, 0
    h, w = mask.shape

    for i in range(1, num_labels):  
        x, y, bw, bh, area = stats[i]
        cx = x + bw // 2
        if area >= min_area:
            if cx < w // 2:
                left_area += area
            else:
                right_area += area

    return left_area > 0 and right_area > 0

#  Estimates actual midline by detecting left and right ventricular regions and computing the mean of their horizontal centroids.
def estimate_actual_midline(mask):

    h, w = mask.shape
    midline_mask = np.zeros_like(mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[1:]  
    if len(stats) < 2:
        return midline_mask

    # Get top 2 largest blobs
    largest = sorted(enumerate(stats), key=lambda x: -x[1][cv2.CC_STAT_AREA])[:2]
    centroids = []
    for i, stat in largest:
        blob = (labels == i + 1).astype(np.uint8)
        M = cv2.moments(blob)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            centroids.append(cx)

    if len(centroids) == 2:
        mid_x = int(np.mean(centroids))
        midline_mask[:, mid_x - 1: mid_x + 2] = 255  

    return midline_mask
