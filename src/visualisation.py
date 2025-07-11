import os
import cv2

def save_visualizations(slice_data_list, output_dir):
    slice_dir = os.path.join(output_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)

    for data in slice_data_list:
        img = cv2.cvtColor(data['image'], cv2.COLOR_GRAY2BGR)
        h = img.shape[0]

        # Draw lines
        cv2.line(img, (data['initial_x'], 0), (data['initial_x'], h-1), (0, 255, 0), 2)  # Green = initial
        cv2.line(img, (data['adjusted_x'], 0), (data['adjusted_x'], h-1), (0, 0, 255), 2)  # Red = adjusted

        out_path = os.path.join(slice_dir, f"slice_{data['index']:03d}.png")
        cv2.imwrite(out_path, img)
