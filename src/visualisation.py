import os
import cv2
import numpy as np

def save_visualizations(slice_data_list, output_dir, ventricle_masks=None):
    import numpy as np
    slice_dir = os.path.join(output_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)

    for i, data in enumerate(slice_data_list):
        img = data['image']
        if img is None:
            print(f"Warning: slice image at index {i} is None, skipping.")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h = img.shape[0]

        if ventricle_masks is not None and i < len(ventricle_masks):
            mask = ventricle_masks[i]

            if mask is None or mask.shape != img.shape[:2]:
                print(f"Skipping overlay for slice {i}: invalid mask.")
            else:
                red_overlay = np.zeros_like(img)
                red_overlay[:, :, 2] = 255
                img = cv2.addWeighted(img, 1.0, red_overlay * (mask[:, :, None] > 0), 0.5, 0)

        # Draw lines
        initial_x = data.get('initial_x')
        adjusted_x = data.get('adjusted_x')
        if initial_x is not None:
            cv2.line(img, (initial_x, 0), (initial_x, h-1), (0, 255, 0), 2)
        if adjusted_x is not None:
            cv2.line(img, (adjusted_x, 0), (adjusted_x, h-1), (0, 0, 255), 2)

        out_path = os.path.join(slice_dir, f"slice_{data['index']:03d}.png")
        cv2.imwrite(out_path, img)
