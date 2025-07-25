from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os, shutil, subprocess, zipfile, uuid, traceback, logging
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt

from src.segmentation import segment_volume_threshold
from src.midline_ideal import compute_ideal_midline_on_slice
from src.midline_actual import estimate_actual_midline_mask_on_slice
from src.mls_calculator import calculate_midline_shift_mm

app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("midline_app")

def save_debug_slice(volume_np, filename):
    mid_idx = volume_np.shape[0] // 2
    mid_slice = volume_np[mid_idx]
    plt.imshow(mid_slice, cmap='gray')
    plt.title(f"Mid Slice ({filename})")
    plt.axis('off')
    os.makedirs("debug_output", exist_ok=True)
    path = os.path.join("debug_output", filename)
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved debug slice: {path}")

def validate_nifti(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    logger.info(f"NIfTI shape: {arr.shape}, min: {arr.min()}, max: {arr.max()}, unique: {np.unique(arr)[:5]}")
    if np.max(arr) == 0:
        raise ValueError(f"NIfTI {path} contains all-zero data. Aborting.")
    return arr

# ——— Utilities ——————————————————————————————————————————————————

def extract_zip_to_temp(uploaded_file):
    temp_id = str(uuid.uuid4())
    extract_dir = f"temp_uploads/{temp_id}"
    os.makedirs(extract_dir, exist_ok=True)

    zip_path = os.path.join(extract_dir, "upload.zip")
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(uploaded_file.file, f)
    logger.info(f"ZIP saved to: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    logger.info(f"Extracted contents: {os.listdir(extract_dir)}")
    return extract_dir

def find_dicom_folder(root_dir):
    for root, _, files in os.walk(root_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            logger.info(f"Found DICOM folder: {root}")
            return root
    raise Exception(f"No .dcm files found in any subfolder of: {root_dir}")

def get_series_with_max_slices(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    logger.info(f"Available SeriesInstanceUIDs: {series_ids}")
    if not series_ids:
        raise Exception(f"No DICOM series found in: {dicom_dir}")

    max_len, selected = 0, None
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
        logger.info(f"Series {sid} → {len(files)} files")
        if len(files) > max_len:
            max_len, selected = len(files), sid

    file_list = reader.GetGDCMSeriesFileNames(dicom_dir, selected)
    logger.info(f"Selected series {selected} with {len(file_list)} slices")
    return file_list
def load_hu_volume(dicom_files):
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    imgs = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    slopes = [float(s.RescaleSlope) for s in slices]
    intercepts = [float(s.RescaleIntercept) for s in slices]

    if len(set(slopes)) > 1 or len(set(intercepts)) > 1:
        logger.warning("Varying slope/intercept in DICOM series")

    volume = imgs * slopes[0] + intercepts[0]
    logger.info(f"Volume shape: {volume.shape}, min: {volume.min()}, max: {volume.max()}")
    return volume

def run_ants_registration(patient_nifti, atlas_nifti, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Starting ANTs registration")

    transform_prefix = os.path.join(out_dir, "transform_")
    dummy = os.path.join(out_dir, "dummy.nii.gz")

    cmd_reg = [
        "antsRegistration",
        "--dimensionality", "3", "--float", "0",
        "--output", f"[{transform_prefix},{dummy}]",
        "--interpolation", "Linear",
        "--use-histogram-matching", "0",
        "--winsorize-image-intensities", "[0.01,0.99]",
        "--initial-moving-transform", f"[{atlas_nifti},{patient_nifti},0]",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{atlas_nifti},{patient_nifti},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox"
    ]
    try:
        logger.info("Running ANTs registration...")
        subprocess.run(cmd_reg, check=True, stderr=open(os.path.join(out_dir, "ants_reg_err.log"), "w"))
    except subprocess.CalledProcessError:
        raise RuntimeError(f"antsRegistration failed. See log: {out_dir}/ants_reg_err.log")

    registered = os.path.join(out_dir, "registered.nii.gz")
    cmd_apply = [
        "antsApplyTransforms", "-d", "3",
        "-i", patient_nifti, "-r", atlas_nifti,
        "-o", registered,
        "-t", transform_prefix + "0GenericAffine.mat",
        "-n", "Linear"
    ]
    try:
        logger.info("Applying transformation...")
        subprocess.run(cmd_apply, check=True, stderr=open(os.path.join(out_dir, "ants_apply_err.log"), "w"))
    except subprocess.CalledProcessError:
        raise RuntimeError(f"antsApplyTransforms failed. See log: {out_dir}/ants_apply_err.log")

    if not os.path.exists(registered):
        raise FileNotFoundError("ANTS registration output missing.")
    logger.info(f"Registration output saved to: {registered}")
    return registered

# ——— FastAPI Endpoint ——————————————————————————————————————————————————

@app.post("/predict_from_dicom")
async def predict_from_dicom(zip_file: UploadFile = File(...)):
    try:
        logger.info("Received new request")

        extract_dir = extract_zip_to_temp(zip_file)
        dicom_root = find_dicom_folder(extract_dir)
        dicom_files = get_series_with_max_slices(dicom_root)
        hu_volume = load_hu_volume(dicom_files)

        # Save original mid-slice for visual debug
        save_debug_slice(hu_volume, "unregistered_input.png")

        sitk_reader = sitk.ImageSeriesReader()
        sitk_reader.SetFileNames(dicom_files)
        series_itk = sitk_reader.Execute()

        os.makedirs("nifti_output", exist_ok=True)
        unreg_nifti = "nifti_output/unregistered.nii.gz"
        hu_img = sitk.GetImageFromArray(hu_volume)
        hu_img.CopyInformation(series_itk)
        sitk.WriteImage(hu_img, unreg_nifti)
        logger.info(f"Saved unregistered HU NIfTI: {unreg_nifti}")

        # Validate NIfTI content before registration
        _ = validate_nifti(unreg_nifti)

        atlas = "atlas/MNI152_T1_1mm.nii.gz"
        registered_nifti = run_ants_registration(unreg_nifti, atlas, "nifti_output")

        registered_np = validate_nifti(registered_nifti)
        save_debug_slice(registered_np, "registered_output.png")

        ref_img = sitk.ReadImage(registered_nifti)
        HU_MIN, HU_MAX = -5, 75
        out_dir = "final_output"
        os.makedirs(out_dir, exist_ok=True)

        masks = segment_volume_threshold(registered_np)

        idx = max(range(masks.shape[0]), key=lambda i: np.count_nonzero(masks[i]))
        logger.info(f"Ideal slice = {idx}")

        _, mid_ideal, _ = compute_ideal_midline_on_slice(
            ct_np=registered_np,
            ideal_slice_index=idx,
            output_dir=out_dir,
            hu_min=HU_MIN,
            hu_max=HU_MAX
        )
        actual_mask = np.zeros_like(masks, dtype=np.uint8)
        actual_mask[idx] = estimate_actual_midline_mask_on_slice(masks[idx])

        ideal_path = os.path.join(out_dir, "ideal.nii.gz")
        actual_path = os.path.join(out_dir, "actual.nii.gz")
        for arr, path in [(mid_ideal, ideal_path), (actual_mask, actual_path)]:
            img = sitk.GetImageFromArray(arr)
            img.CopyInformation(ref_img)
            sitk.WriteImage(img, path)
            logger.info(f"Saved mask: {path}")

        _, avg_shift = calculate_midline_shift_mm(
            ideal_midline_path=ideal_path,
            actual_midline_path=actual_path,
            output_path=os.path.join(out_dir, "shifts.npz")
        )
        logger.info(f"Midline shift = {avg_shift:.2f} mm")

        return {
            "ideal_slice_index": idx,
            "midline_shift_mm": float(avg_shift),
            "debug_images": {
                "unregistered_input": "debug_output/unregistered_input.png",
                "registered_output": "debug_output/registered_output.png"
            }
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error in /predict_from_dicom:\n" + tb)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})