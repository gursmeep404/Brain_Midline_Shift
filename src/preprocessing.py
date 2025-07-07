import numpy as np
import cv2


def load_dicom_series(dicom_folder):
    import pydicom
    import os
    from collections import defaultdict

    # Group files by SeriesInstanceUID
    series_dict = defaultdict(list)
    for f in os.listdir(dicom_folder):
        if not f.endswith(".dcm"):
            continue
        path = os.path.join(dicom_folder, f)
        dcm = pydicom.dcmread(path)
        uid = dcm.SeriesInstanceUID
        series_dict[uid].append(dcm)

    # Choose series with most slices
    best_series = max(series_dict.values(), key=len)

    # Sort by slice location
    best_series.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    slices = []
    for dcm in best_series:
        hu = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        slices.append(hu)

    volume = np.stack(slices).astype(np.float32)
    return volume


def normalize_volume(volume, min_hu=-100, max_hu=100):
    clipped = np.clip(volume, min_hu, max_hu)
    norm = ((clipped - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
    return norm


def resize_volume(volume, shape=(30, 256, 256)):
    from skimage.transform import resize
    return resize(volume, shape, preserve_range=True).astype(np.uint8)