import os
import cv2
import numpy as np

def save_visualizations(slice_data_list, output_dir, ventricle_masks=None, actual_midline_data=None):
    slice_dir = os.path.join(output_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)

    for i, data in enumerate(slice_data_list):
        img = data['image']
        if img is None:
            print(f"Warning: slice image at index {i} is None, skipping.")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  
        h = img.shape[0]

        # Overlay ventricle mask 
        if ventricle_masks is not None and i < len(ventricle_masks):
            mask = ventricle_masks[i]
            if mask is not None and mask.shape == img.shape[:2]:
                red_overlay = np.zeros_like(img)
                red_overlay[:, :, 2] = 255  # Red channel
                img = cv2.addWeighted(img, 1.0, red_overlay * (mask[:, :, None] > 0), 0.5, 0)

        # ideal midline (red)
        adjusted_x = data.get('adjusted_x')
        if adjusted_x is not None:
            cv2.line(img, (adjusted_x, 0), (adjusted_x, h - 1), (0, 0, 255), 2)  # red line
        else:
            print(f"No adjusted_x found for slice {i}")

        # actual midline (blue)
        if actual_midline_data is not None:
            
            if isinstance(actual_midline_data, dict):
                actual_slice = actual_midline_data if actual_midline_data.get("index") == i else None
            else:
                actual_slice = next((s for s in actual_midline_data if s["index"] == i), None)

            if actual_slice:
                actual_mid_x = actual_slice.get("actual_mid_x")
                if actual_mid_x is not None:
                    print(f"Drawing actual midline at x={actual_mid_x} for slice {i}")
                    cv2.line(img, (actual_mid_x, 0), (actual_mid_x, h - 1), (255, 0, 0), 2)  # blue line
                else:
                    print(f"actual_mid_x missing for slice {i}")
            else:
                print(f"No actual_slice found for index {i}")

        out_path = os.path.join(slice_dir, f"slice_{data['index']:03d}.png")
        cv2.imwrite(out_path, img)
