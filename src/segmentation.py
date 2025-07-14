import numpy as np
import cv2

#  Segment ventricular regions based on HU range. Returns a binary 2D mask.
def segment_ventricles(slice_img, hu_low=-5, hu_high=20, min_area=100):

    # 1. Threshold to isolate CSF-like HU range
    mask = np.logical_and(slice_img >= hu_low, slice_img <= hu_high).astype(np.uint8) * 255

    # 2. Morphological cleaning
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # 3. Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255

    return filtered_mask

#   Loop over slices in the 3D volume and apply segmentation. Returns a 3D binary mask with the same shape.
def segment_volume_threshold(volume, hu_low=-5, hu_high=20, min_area=100):
    
    masks = np.zeros_like(volume, dtype=np.uint8)
    for i in range(volume.shape[0]):
        masks[i] = segment_ventricles(volume[i], hu_low=hu_low, hu_high=hu_high, min_area=min_area)
    return masks
