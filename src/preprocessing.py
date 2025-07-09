import os
import numpy as np
import pydicom
from collections import defaultdict
from skimage.transform import resize

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

    # Choose the series with the most slices
    best_series = max(series_dict.values(), key=len)
    print(f"Selected series with {len(best_series)} slices")

    # Sort slices by Z-location or InstanceNumber
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


# Convert HU volume to 8-bit grayscale by clipping and scaling.
def normalize_volume(volume, min_hu=-100, max_hu=100):

    volume = np.clip(volume, min_hu, max_hu)
    norm_volume = ((volume - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
    return norm_volume

