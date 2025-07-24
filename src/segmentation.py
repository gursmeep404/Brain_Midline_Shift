import numpy as np
import cv2

def segment_ventricles(slice_img, hu_low=-5, hu_high=20, min_area=100):
    # Threshold to isolate CSF-like HU range
    mask = np.logical_and(slice_img >= hu_low, slice_img <= hu_high).astype(np.uint8) * 255

    # Morphological cleaning
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Removing small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255

    return filtered_mask

# Remove 20% slices from start & end. Remove top 1/3rd and bottom 1/3rd of segmentation in each slice
def segment_volume_threshold(volume, hu_low=-5, hu_high=20, min_area=100, trim_percent=0.2):
    masks = np.zeros_like(volume, dtype=np.uint8)
    num_slices, h, w = volume.shape

    for i in range(num_slices):
        mask = segment_ventricles(volume[i], hu_low=hu_low, hu_high=hu_high, min_area=min_area)

        # **Black out top 1/3rd**
        top_cut = h // 3
        mask[:top_cut, :] = 0

        # **Black out bottom 1/3rd**
        bottom_cut = 2 * h // 3
        mask[bottom_cut:, :] = 0

        masks[i] = mask

    # **Remove 20% from start and 20% from end**
    trim_slices = int(num_slices * trim_percent)
    masks[:trim_slices] = 0
    masks[-trim_slices:] = 0

    return masks
