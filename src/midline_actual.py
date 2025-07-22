import numpy as np
import cv2
import os

_actual_midline_data = {}

def extract_shape_features(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu_features = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)  # log scale
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

def match_blob_to_templates(blob, template_features, threshold=2.0):
    feats = extract_shape_features(blob)
    for f1 in feats:
        for f2 in template_features:
            dist = np.linalg.norm(f1 - f2)
            if dist < threshold:
                return True
    return False




def estimate_actual_midline_mask(ventricle_masks, template_dir, min_area=100):
    global _actual_midline_data
    
    h, w = ventricle_masks.shape[1:]
    midline_volume = np.zeros_like(ventricle_masks)

    template_features = load_template_features(template_dir)

    mid_xs = []
    valid_slices = []
    slice_data_list = []  # Store per-slice midline positions

    for i in range(ventricle_masks.shape[0]):
        mask = ventricle_masks[i].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [cnt], -1, 255, -1)
            M = cv2.moments(blob)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            blobs.append((cx, blob))

        if len(blobs) < 2:
            continue

        left_blobs = [b for b in blobs if b[0] < w // 2]
        right_blobs = [b for b in blobs if b[0] >= w // 2]

        best_mid_x = None
        for lc in left_blobs:
            for rc in right_blobs:
                # Relaxed distance constraint
                if abs(lc[0] - rc[0]) < w * 0.15:
                    continue  

                left_blob = lc[1]
                right_blob = rc[1]

                left_match = match_blob_to_templates(left_blob, template_features)
                right_match = match_blob_to_templates(right_blob, template_features)

                # Accept if at least one matches
                if left_match or right_match:
                    mid_x = int((lc[0] + rc[0]) / 2)
                    best_mid_x = mid_x
                    break

            if best_mid_x is not None:
                break

        # Fallback if no template match but both blobs are present
        if best_mid_x is None and left_blobs and right_blobs:
            lc = max(left_blobs, key=lambda b: cv2.countNonZero(b[1]))  # largest left
            rc = max(right_blobs, key=lambda b: cv2.countNonZero(b[1]))  # largest right
            best_mid_x = int((lc[0] + rc[0]) / 2)


        if best_mid_x is not None:
            mid_xs.append(best_mid_x)
            valid_slices.append(i)
            slice_data_list.append({"index": i, "actual_mid_x": best_mid_x})

    if len(mid_xs) > 0:
        final_mid_x = int(np.median(mid_xs))
        for i in valid_slices:
            midline_volume[i, :, final_mid_x - 1:final_mid_x + 2] = 255

    # Store extra data for later visualization
    _actual_midline_data["slice_data_list"] = slice_data_list

    return midline_volume

def get_actual_midline_data():
    return _actual_midline_data.get("slice_data_list", [])
