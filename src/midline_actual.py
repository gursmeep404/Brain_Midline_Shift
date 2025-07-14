import numpy as np
import cv2
import os

def extract_shape_features(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu_features = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)  
        hu_features.append(hu)
    return hu_features

def load_template_features(template_dir):
    template_features = []
    for fname in os.listdir(template_dir):
        if not fname.endswith('.png'):
            continue
        path = os.path.join(template_dir, fname)
        tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        tmpl = cv2.threshold(tmpl, 127, 255, cv2.THRESH_BINARY)[1]
        features = extract_shape_features(tmpl)
        template_features.extend(features)
    return template_features

def match_shapes(segmented_features, template_features, threshold=0.6):
    for f1 in segmented_features:
        for f2 in template_features:
            dist = np.linalg.norm(f1 - f2)
            if dist < threshold:
                return True
    return False

def estimate_actual_midline_mask(ventricle_masks, template_dir, min_area=100):
    """
    Args:
        ventricle_masks: 3D numpy array (segmented masks, same shape as volume)
        template_dir: folder containing 2D ventricle template PNGs
    Returns:
        midline_volume: 3D numpy array with vertical line at estimated midline
    """
    h, w = ventricle_masks.shape[1:]
    midline_volume = np.zeros_like(ventricle_masks)

    template_feats = load_template_features(template_dir)

    for i in range(ventricle_masks.shape[0]):
        mask = ventricle_masks[i].astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            blob_mask = np.zeros_like(mask)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            blobs.append(blob_mask)

        if len(blobs) < 2:
            continue

        combined_mask = np.zeros_like(mask)
        for b in blobs:
            combined_mask = cv2.bitwise_or(combined_mask, b)

        feats = extract_shape_features(combined_mask)
        if not match_shapes(feats, template_feats):
            continue  

        # Compute centroids of two largest blobs
        stats = [cv2.moments(b) for b in blobs]
        cx = []
        for M in stats:
            if M['m00'] != 0:
                cx.append(int(M['m10'] / M['m00']))

        if len(cx) >= 2:
            mid_x = int(np.mean(sorted(cx)[:2]))
            midline_volume[i, :, mid_x-1:mid_x+2] = 255

    return midline_volume
