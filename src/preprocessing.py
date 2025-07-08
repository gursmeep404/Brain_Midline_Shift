import numpy as np
import cv2

def load_dicom_series(dicom_folder):
    import pydicom
    import os
    import numpy as np
    from collections import defaultdict

    # Group DICOMs by SeriesInstanceUID
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
        raise ValueError("No valid DICOM series found in the folder.")

    # Choose series with the most slices
    best_series = max(series_dict.values(), key=len)
    print(f"Selected series with {len(best_series)} slices")

    # Sort slices based on Z-position or fallback to InstanceNumber
    try:
        best_series.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    except:
        best_series.sort(key=lambda d: int(d.InstanceNumber))

    # Convert each slice to HU (Hounsfield Units)
    slices = []
    for dcm in best_series:
        array = dcm.pixel_array.astype(np.float32)
        slope = getattr(dcm, "RescaleSlope", 1)
        intercept = getattr(dcm, "RescaleIntercept", 0)
        hu = array * slope + intercept
        slices.append(hu)

    # Stack slices into 3D volume
    volume = np.stack(slices, axis=0).astype(np.float32)
    return volume



def normalize_volume(volume, min_hu=-100, max_hu=100):
    clipped = np.clip(volume, min_hu, max_hu)
    norm = ((clipped - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
    return norm

def resize_volume(volume, target_hw=(256, 256)):
    from skimage.transform import resize
    num_slices = volume.shape[0] 
    resized_slices = []

    for i in range(num_slices):
        resized = resize(volume[i], target_hw, preserve_range=True, anti_aliasing=True)
        resized_slices.append(resized.astype(np.uint8))

    resized_volume = np.stack(resized_slices, axis=0)
    return resized_volume
