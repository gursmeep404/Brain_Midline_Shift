import os
import numpy as np
import pydicom
from collections import defaultdict
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

def load_dicom_series(dicom_folder):
    series_dict = defaultdict(list)
    for f in os.listdir(dicom_folder):
        if not f.lower().endswith(".dcm"):
            continue
        path = os.path.join(dicom_folder, f)
        try:
            dcm = pydicom.dcmread(path)
            uid = dcm.SeriesInstanceUID
            series_dict[uid].append(dcm)
        except Exception as e:
            print(f"Skipping file {path}: {e}")

    if not series_dict:
        raise ValueError("No valid DICOM series found.")

    best_series = max(series_dict.values(), key=len)
    print(f"Selected series with {len(best_series)} slices")

    # Sort slices by position or instance number
    try:
        best_series.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    except:
        best_series.sort(key=lambda d: int(d.InstanceNumber))

    slices = []
    for dcm in best_series:
        img = dcm.pixel_array.astype(np.float32)
        slope = getattr(dcm, "RescaleSlope", 1)
        intercept = getattr(dcm, "RescaleIntercept", 0)
        hu_img = img * slope + intercept
        slices.append(hu_img)

    volume = np.stack(slices, axis=0)
    return volume

# Skull stripping by removing high HU values 
def skull_strip(volume, hu_threshold=300):
    stripped = np.copy(volume)
    stripped[volume > hu_threshold] = 0
    return stripped

# Normalize and convert HU volume to 8-bit grayscale
def normalize_volume(volume, hu_min=0, hu_max=80):
    volume = np.clip(volume, hu_min, hu_max)
    norm_volume = ((volume - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
    return norm_volume

# Resize volume to uniform shape (e.g., 256x256 per slice)
def resize_volume(volume, output_shape=(256, 256)):
    resized_slices = [resize(slice, output_shape, mode='constant', preserve_range=True).astype(np.uint8)
                      for slice in volume]
    return np.stack(resized_slices, axis=0)

# Apply Gaussian filter to reduce noise
def apply_smoothing(volume, sigma=1.0):
    return gaussian_filter(volume, sigma=(0, 1, 1))  # smooth x and y only, not z

# Full preprocessing pipeline
def preprocess_dicom(dicom_folder,
                     hu_min=0,
                     hu_max=80,
                     resize_shape=None,
                     smooth_sigma=None):
    
    print("Loading DICOM series...")
    volume = load_dicom_series(dicom_folder)

    print("Skull stripping...")
    stripped_volume = skull_strip(volume)

    print(f"Normalizing HU to grayscale (HU range {hu_min}-{hu_max})...")
    norm_volume = normalize_volume(stripped_volume, hu_min=hu_min, hu_max=hu_max)

    if resize_shape:
        print(f"Resizing to shape {resize_shape}...")
        norm_volume = resize_volume(norm_volume, resize_shape)

    if smooth_sigma:
        print(f"Applying Gaussian smoothing (sigma={smooth_sigma})...")
        norm_volume = apply_smoothing(norm_volume, sigma=smooth_sigma)

    print("Preprocessing complete.")
    return norm_volume
