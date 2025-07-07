import numpy as np
def estimate_actual_midline(ventricle_mask):
    y, x = np.where(ventricle_mask > 0)
    if len(x) == 0:
        return None
    return int(np.mean(x))
