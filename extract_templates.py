import os
import numpy as np
import nibabel as nib
import cv2

def extract_ventricle_templates(
    csf_path,
    output_dir="templates",
    threshold=0.5,
    area_threshold=100,
    slice_fraction=0.4,
):
    """
    Extract 2D ventricle templates from a CSF probability map in a brain atlas.

    Args:
        csf_path: Path to mni_icbm152_csf_tal_nlin_sym_09c.nii
        output_dir: Folder where PNGs will be saved
        threshold: CSF probability threshold (default=0.5)
        area_threshold: Minimum component area to retain (in pixels)
        slice_fraction: Central fraction of axial slices to use (e.g., 0.4 = middle 40%)
    """
    print(f"[INFO] Loading CSF atlas from: {csf_path}")
    csf_img = nib.load(csf_path)
    csf_data = csf_img.get_fdata()

    print("[INFO] Binarizing CSF probability volume...")
    csf_bin = (csf_data > threshold).astype(np.uint8)

    total_slices = csf_bin.shape[2]
    start = int(total_slices * (0.5 - slice_fraction / 2))
    end = int(total_slices * (0.5 + slice_fraction / 2))

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Extracting from axial slices {start} to {end} (middle {slice_fraction*100:.0f}%)...")

    count = 0
    for i in range(start, end):
        raw_mask = csf_bin[:, :, i] * 255
        raw_mask = raw_mask.astype(np.uint8)

        # Rotate for upright radiological view (anterior up)
        rotated_mask = cv2.rotate(raw_mask, cv2.ROTATE_90_CLOCKWISE)
        rotated_mask = cv2.flip(rotated_mask, 0)

        # Remove small blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rotated_mask)
        clean_mask = np.zeros_like(rotated_mask)

        for j in range(1, num_labels):  # skip background
            area = stats[j, cv2.CC_STAT_AREA]
            if area >= area_threshold:
                clean_mask[labels == j] = 255

        if np.count_nonzero(clean_mask) == 0:
            continue

        filename = f"template_{count:03d}.png"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, clean_mask)
        count += 1

    print(f"[DONE] Saved {count} ventricle template images to: {output_dir}")

if __name__ == "__main__":
    csf_path = "atlas/mni_icbm152_nlin_sym_09c_nifti/mni_icbm152_nlin_sym_09c/mni_icbm152_csf_tal_nlin_sym_09c.nii"
    output_dir = "templates"
    extract_ventricle_templates(csf_path, output_dir)
