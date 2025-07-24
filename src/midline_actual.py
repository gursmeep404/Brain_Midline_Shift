import numpy as np
import cv2
import os

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
    for lbl in range(1, num_labels):  
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
        return (int(round(cx)), 0), (int(round(cx)), h - 1)

    t_min = min(ts)
    t_max = max(ts)

    x1 = cx + t_min * vx
    y1 = cy + t_min * vy
    x2 = cx + t_max * vx
    y2 = cy + t_max * vy

    return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))


#For visualisation
def get_actual_midline_data(volume, slice_index, hu_min=-5, hu_max=75):
    if slice_index < 0 or slice_index >= volume.shape[0]:
        raise IndexError(f"Slice index {slice_index} is out of range (0 to {volume.shape[0] - 1})")

    slice_hu = volume[slice_index]
    gray = np.clip(slice_hu, hu_min, hu_max)
    gray_norm = ((gray - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)

    data = {
        "index": slice_index,
        "image": gray_norm
    }

    return data

def estimate_actual_midline_mask_on_slice(mask_slice, min_area=100,
                                          morph_kernel=3, keep_top=2, thickness=3,
                                          draw_mode="pc1"):
    h, w = mask_slice.shape
    cleaned = _clean_ventricle_slice(mask_slice, min_area=min_area,
                                     morph_kernel=morph_kernel,
                                     keep_top=keep_top)

    ys, xs = np.nonzero(cleaned)
    midline_mask = np.zeros_like(mask_slice, dtype=np.uint8)

    if xs.size < 5:
        # Not enough pixels to compute PCA then return empty mask
        return midline_mask  

    (vx, vy), (cx, cy) = _pca_pc1_direction(xs, ys)

    if draw_mode == "vertical":
        mid_x = int(round(cx))
        x1 = max(0, mid_x - thickness // 2)
        x2 = min(w, mid_x + (thickness + 1) // 2)
        midline_mask[:, x1:x2] = 255
    else:
        p1, p2 = _clip_line_to_image(cx, cy, vx, vy, w, h)
        cv2.line(midline_mask, p1, p2, color=255, thickness=thickness)

    return midline_mask
