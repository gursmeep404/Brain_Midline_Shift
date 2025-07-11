import os
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage

# CONFIGURATION
NIFTI_PATH = "patient_trial.nii.gz" 
OUTPUT_DIR = "output_trial"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HU thresholds for bone detection
BONE_HU_MIN = 300  # Minimum HU value for bone
BONE_HU_MAX = 2000  # Maximum HU value for bone


def normalize_to_uint8(slice_np, hu_min=-1000, hu_max=1000):
    """Convert HU values to 8-bit grayscale with proper windowing."""
    slice_np = np.clip(slice_np, hu_min, hu_max)
    normalized = ((slice_np - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
    return normalized

def get_optimal_window(slice_hu):
    """Get optimal window/level for brain CT."""
    # For brain CT, typical windows:
    # Brain window: 80 HU width, 40 HU center -> -40 to 40 HU
    # Bone window: 2000 HU width, 400 HU center -> -600 to 1400 HU
    
    # Calculate percentiles to avoid outliers
    p1, p99 = np.percentile(slice_hu, [1, 99])
    
    # Use brain window if most values are in brain range
    if p99 < 200:  # Mostly soft tissue
        return -40, 120  # Brain window
    else:  # Has bone
        return -200, 1200  # Wider window to show both brain and bone


def generate_bone_mask(slice_hu):
    """Generate bone mask using HU thresholding for skull detection."""
    mask = ((slice_hu >= BONE_HU_MIN) & (slice_hu <= BONE_HU_MAX)).astype(np.uint8)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply median blur to smooth the mask
    mask = cv2.medianBlur(mask, 3)
    
    return mask


def compute_midline(mask):
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return mask.shape[1] // 2
    return int(np.mean(x))


def draw_midline(slice_gray, mid_x, color=(0, 255, 0)):
    """Draw the midline on the slice image."""
    vis = cv2.cvtColor(slice_gray, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (mid_x, 0), (mid_x, slice_gray.shape[0]-1), color, 2)
    return vis

def visualize_slice_with_midline(slice_hu, bone_mask, mid_x, slice_idx):
    """Create a visualization showing the original slice, bone mask, and midline."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original slice
    axes[0].imshow(slice_hu, cmap='gray')
    axes[0].axvline(x=mid_x, color='red', linewidth=2, label='Midline')
    axes[0].set_title(f'Original Slice {slice_idx}')
    axes[0].legend()
    
    # Bone mask
    axes[1].imshow(bone_mask, cmap='gray')
    axes[1].axvline(x=mid_x, color='red', linewidth=2, label='Midline')
    axes[1].set_title(f'Bone Mask {slice_idx}')
    axes[1].legend()
    
    # Overlay
    axes[2].imshow(slice_hu, cmap='gray')
    axes[2].imshow(bone_mask, cmap='Reds', alpha=0.3)
    axes[2].axvline(x=mid_x, color='red', linewidth=2, label='Midline')
    axes[2].set_title(f'Overlay {slice_idx}')
    axes[2].legend()
    
    plt.tight_layout()
    return fig


def compute_center_of_mass(mask):
    """Compute the center of mass of the bone mask (skull) to determine midline."""
    if np.sum(mask) == 0:
        return mask.shape[1] // 2
    
    # Calculate center of mass for the entire mask
    center_y, center_x = ndimage.center_of_mass(mask)
    
    # Return the x-coordinate as the midline position
    return int(center_x)

def main():
    print("Loading NIfTI file...")
    if not os.path.exists(NIFTI_PATH):
        print(f"Error: NIfTI file not found at {NIFTI_PATH}")
        print("Please update the NIFTI_PATH variable with the correct path to your patient file.")
        return
        
    ct_volume = sitk.ReadImage(NIFTI_PATH)
    print(f"\n=== VOLUME INFORMATION ===")
    print(f"Volume size (X,Y,Z): {ct_volume.GetSize()}")
    print(f"Volume spacing: {ct_volume.GetSpacing()}")
    print(f"Volume origin: {ct_volume.GetOrigin()}")
    print(f"Pixel type: {ct_volume.GetPixelIDTypeAsString()}")
    spacing = ct_volume.GetSpacing()
    print("Spacing:", spacing)

    
    # Convert to numpy and check properties
    ct_np = sitk.GetArrayFromImage(ct_volume)
    print(f"\n=== NUMPY ARRAY INFORMATION ===")
    print(f"NumPy shape (Z,Y,X): {ct_np.shape}")
    print(f"Data type: {ct_np.dtype}")
    print(f"HU value range: {ct_np.min():.1f} to {ct_np.max():.1f}")
    print(f"Mean HU: {ct_np.mean():.1f}")
    
    # Check a few sample slices
    mid_slice = ct_np.shape[0] // 2
    print(f"\n=== SAMPLE SLICE ANALYSIS ===")
    print(f"Middle slice ({mid_slice}) HU range: {ct_np[mid_slice].min():.1f} to {ct_np[mid_slice].max():.1f}")
    
    # Check for empty/mostly empty slices
    non_zero_counts = []
    for i in range(0, ct_np.shape[0], ct_np.shape[0]//10):  # Check every 10th slice
        non_zero = np.count_nonzero(ct_np[i] != ct_np[i].min())
        non_zero_counts.append((i, non_zero))
        print(f"Slice {i}: {non_zero} non-background pixels")


    print("Processing slices...")
    ct_np = sitk.GetArrayFromImage(ct_volume)  # shape: [slices, height, width]
    
    # Create directories for different outputs
    os.makedirs(os.path.join(OUTPUT_DIR, "slices"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

    # Process middle slices for better visualization (skull is more visible)
    start_slice = max(0, len(ct_np) // 4)
    end_slice = min(len(ct_np), 3 * len(ct_np) // 4)
    
    for i in range(start_slice, end_slice):
        slice_hu = ct_np[i]
        
        # Check if slice has meaningful content
        non_background = np.count_nonzero(slice_hu != slice_hu.min())
        if non_background < 1000:  # Skip mostly empty slices
            print(f"Skipping slice {i}: only {non_background} non-background pixels")
            continue
            
        # Use optimal windowing for this slice
        hu_min, hu_max = get_optimal_window(slice_hu)
        gray = normalize_to_uint8(slice_hu, hu_min, hu_max)
        
        print(f"Processing slice {i}: HU range {slice_hu.min():.1f} to {slice_hu.max():.1f}, windowed {hu_min} to {hu_max}")

        bone_mask = generate_bone_mask(slice_hu)
        mid_x = compute_center_of_mass(bone_mask)

        # Save slice with midline
        output_img = draw_midline(gray, mid_x)
        out_path = os.path.join(OUTPUT_DIR, "slices", f"slice_{i:03d}.png")
        cv2.imwrite(out_path, output_img)
        
        # Create and save visualization for every 10th slice
        if i % 10 == 0:
            fig = visualize_slice_with_midline(slice_hu, bone_mask, mid_x, i)
            vis_path = os.path.join(OUTPUT_DIR, "visualizations", f"visualization_{i:03d}.png")
            fig.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"Done. Processed {end_slice - start_slice} slices.")
    print(f"- Slice images saved to '{OUTPUT_DIR}/slices'")
    print(f"- Visualizations saved to '{OUTPUT_DIR}/visualizations'")
    print(f"\nTo run the script, update NIFTI_PATH to point to your patient file.")


if __name__ == "__main__":
    main()
