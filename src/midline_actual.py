import numpy as np
import cv2
import os

_actual_midline_data = {}

# Mask cleaning using flood fill to fill the internal holes in the mask
def _fill_holes(mask01):
   
    h, w = mask01.shape
    flood = mask01.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=1)
    holes = 1 - flood
    return (mask01 | holes).astype(np.uint8)


def _clean_ventricle_slice(mask_slice, min_area=100, morph_kernel=3, keep_top=2):

    m = (mask_slice > 0).astype(np.uint8)

    if morph_kernel and morph_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    m = _fill_holes(m)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return (m * 255).astype(np.uint8)

    comps = []
    for lbl in range(1, num_labels):  # skip background
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            comps.append((area, lbl))

    if not comps:
        return np.zeros_like(m, dtype=np.uint8)

    comps.sort(reverse=True)
    keep = [lbl for _, lbl in comps[:keep_top]]

    out = np.zeros_like(m, dtype=np.uint8)
    for lbl in keep:
        out[labels == lbl] = 1

    return (out * 255).astype(np.uint8)



def _pca_pc1_direction(xs, ys):
   
    
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    # center
    mean = pts.mean(axis=0, keepdims=True)
    pts_c = pts - mean
    # covariance
    cov = np.cov(pts_c, rowvar=False)
    
    eigvals, eigvecs = np.linalg.eigh(cov)  
    idx = np.argmax(eigvals) 
    pc1 = eigvecs[:, idx]    
    # normalize
    n = np.linalg.norm(pc1)
    if n > 0:
        pc1 = pc1 / n
    return pc1, mean.ravel() 


def _clip_line_to_image(cx, cy, vx, vy, w, h):
    
    eps = 1e-8
    ts = []

    # Intersections with vertical borders x=0 and x=w-1
    if abs(vx) > eps:
        t = (0 - cx) / vx
        y = cy + t * vy
        if 0 <= y <= h - 1:
            ts.append(t)
        t = (w - 1 - cx) / vx
        y = cy + t * vy
        if 0 <= y <= h - 1:
            ts.append(t)

    # Intersections with horizontal borders y=0 and y=h-1
    if abs(vy) > eps:
        t = (0 - cy) / vy
        x = cx + t * vx
        if 0 <= x <= w - 1:
            ts.append(t)
        t = (h - 1 - cy) / vy
        x = cx + t * vx
        if 0 <= x <= w - 1:
            ts.append(t)

    if len(ts) < 2:
        # fallback: degenerate direction; produce a short vertical tick
        return (int(round(cx)), 0), (int(round(cx)), h - 1)

    t_min = min(ts)
    t_max = max(ts)

    x1 = cx + t_min * vx
    y1 = cy + t_min * vy
    x2 = cx + t_max * vx
    y2 = cy + t_max * vy

    return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))


def estimate_actual_midline_mask(ventricle_masks, template_dir=None, min_area=100,
                                 morph_kernel=3, keep_top=2, thickness=3,
                                 draw_mode="pc1"):
   
    global _actual_midline_data

    n_slices, h, w = ventricle_masks.shape
    midline_volume = np.zeros((n_slices, h, w), dtype=np.uint8)

    slice_records = []

    for i in range(n_slices):
        raw = ventricle_masks[i]

        # Clean slice mask
        cleaned = _clean_ventricle_slice(raw, min_area=min_area,
                                         morph_kernel=morph_kernel,
                                         keep_top=keep_top)

        area = int(np.count_nonzero(cleaned))
        if area < min_area:
            # Not enough ventricle pixels to trust
            slice_records.append({
                "index": i,
                "usable": False,
                "reason": "area<min_area",
                "actual_mid_x": None,
                "pc1_vx": None,
                "pc1_vy": None,
                "centroid_x": None,
                "centroid_y": None,
                "area": area
            })
            continue

        ys, xs = np.nonzero(cleaned)
        if xs.size < 5:
            slice_records.append({
                "index": i,
                "usable": False,
                "reason": "too_few_pixels",
                "actual_mid_x": None,
                "pc1_vx": None,
                "pc1_vy": None,
                "centroid_x": None,
                "centroid_y": None,
                "area": area
            })
            continue

        (vx, vy), (cx, cy) = _pca_pc1_direction(xs, ys)  # PC1 + centroid

        if draw_mode == "vertical":
            # Force vertical line at centroid x (robust for MLS)
            mid_x = int(round(cx))
            x1 = max(0, mid_x - thickness // 2)
            x2 = min(w, mid_x + (thickness + 1) // 2)
            midline_volume[i, :, x1:x2] = 255
            used_mid_x = mid_x

        else:  # "pc1": draw actual PC1 line through centroid
            p1, p2 = _clip_line_to_image(cx, cy, vx, vy, w, h)
            cv2.line(midline_volume[i], p1, p2, color=255, thickness=thickness)
            # For MLS: use centroid x
            used_mid_x = int(round(cx))

        slice_records.append({
            "index": i,
            "usable": True,
            "reason": "ok",
            "actual_mid_x": used_mid_x,
            "pc1_vx": float(vx),
            "pc1_vy": float(vy),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "area": area
        })

    # Store per-slice info for visualization / debugging
    _actual_midline_data["slice_data_list"] = slice_records

    return midline_volume

# For visualisation
def get_actual_midline_data():
  
    return _actual_midline_data.get("slice_data_list", [])

# import numpy as np
# import cv2
# import os

# _actual_midline_data = {}


# def _clean_ventricle_slice(mask_slice, min_area=100, morph_kernel=3, keep_top=2):
#     """
#     Basic cleaning of each slice to keep only the main ventricle regions.
#     """
#     m = (mask_slice > 0).astype(np.uint8)

#     if morph_kernel and morph_kernel > 1:
#         k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
#         m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
#         m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

#     # Remove small connected components
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
#     if num_labels <= 1:
#         return (m * 255).astype(np.uint8)

#     comps = [(stats[lbl, cv2.CC_STAT_AREA], lbl) for lbl in range(1, num_labels)]
#     comps = [c for c in comps if c[0] >= min_area]
#     if not comps:
#         return np.zeros_like(m, dtype=np.uint8)

#     comps.sort(reverse=True)
#     keep = [lbl for _, lbl in comps[:keep_top]]
#     out = np.isin(labels, keep).astype(np.uint8)
#     return (out * 255).astype(np.uint8)


# def _compute_centroid(mask):
#     """
#     Compute the centroid (center of mass) of a binary mask.
#     """
#     M = cv2.moments(mask)
#     if M["m00"] == 0:
#         return None
#     cx = int(M["m10"] / M["m00"])
#     cy = int(M["m01"] / M["m00"])
#     return cx, cy


# def estimate_actual_midline_mask(ventricle_masks, template_dir=None, min_area=100, morph_kernel=3, keep_top=2, thickness=3):
#     """
#     Estimate actual midline by drawing a vertical line through the centroid of ventricles in each slice.
#     """
#     global _actual_midline_data

#     n_slices, h, w = ventricle_masks.shape
#     midline_volume = np.zeros((n_slices, h, w), dtype=np.uint8)
#     slice_records = []

#     for i in range(n_slices):
#         raw = ventricle_masks[i]
#         cleaned = _clean_ventricle_slice(raw, min_area=min_area, morph_kernel=morph_kernel, keep_top=keep_top)

#         area = int(np.count_nonzero(cleaned))
#         if area < min_area:
#             slice_records.append({
#                 "index": i, "usable": False, "reason": "area<min_area", "actual_mid_x": None
#             })
#             continue

#         centroid = _compute_centroid(cleaned)
#         if centroid is None:
#             slice_records.append({
#                 "index": i, "usable": False, "reason": "no_centroid", "actual_mid_x": None
#             })
#             continue

#         cx, cy = centroid
#         x1 = max(0, cx - thickness // 2)
#         x2 = min(w, cx + (thickness + 1) // 2)
#         midline_volume[i, :, x1:x2] = 255

#         slice_records.append({
#             "index": i, "usable": True, "reason": "ok", "actual_mid_x": cx,
#             "centroid_x": float(cx), "centroid_y": float(cy), "area": area
#         })

#     _actual_midline_data["slice_data_list"] = slice_records
#     return midline_volume


# def get_actual_midline_data():
#     return _actual_midline_data.get("slice_data_list", [])
