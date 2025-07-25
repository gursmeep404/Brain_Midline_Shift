from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os, shutil, subprocess, zipfile, uuid, traceback, logging
import numpy as np
import pydicom
import SimpleITK as sitk

from src.segmentation import segment_volume_threshold
from src.midline_ideal import compute_ideal_midline_on_slice
from src.midline_actual import estimate_actual_midline_mask_on_slice
from src.mls_calculator import calculate_midline_shift_mm

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("midline_app")


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
    if len(set(slopes)) > 1:
        logger.warning("Different RescaleSlope values across slices")
    if len(set(intercepts)) > 1:
        logger.warning("Different RescaleIntercept values across slices")

    return imgs * slopes[0] + intercepts[0]


def run_ants_registration(patient_nifti, atlas_nifti, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Starting ANTs registration")

    transform_prefix = os.path.join(out_dir, "transform_")
    dummy = os.path.join(out_dir, "dummy.nii.gz")
    cmd_reg = [
        "antsRegistration",
        "--dimensionality", "3",
        "--output", f"[{transform_prefix},{dummy}]",
        "--initial-moving-transform", f"[{atlas_nifti},{patient_nifti},1]",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{atlas_nifti},{patient_nifti},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox"
    ]
    logger.info("ANTS cmd: " + " ".join(cmd_reg))
    subprocess.run(cmd_reg, check=True)

    registered = os.path.join(out_dir, "registered.nii.gz")
    cmd_apply = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", patient_nifti,
        "-r", atlas_nifti,
        "-o", registered,
        "-t", transform_prefix + "0GenericAffine.mat"
    ]
    logger.info("ANTS apply cmd: " + " ".join(cmd_apply))
    subprocess.run(cmd_apply, check=True)

    logger.info(f"Registration output: {registered}")
    return registered


# ——— FastAPI Endpoint ——————————————————————————————————————————————————

@app.post("/predict_from_dicom")
async def predict_from_dicom(zip_file: UploadFile = File(...)):
    try:
        logger.info("Received new request")

        # 1. Extract ZIP
        extract_dir = extract_zip_to_temp(zip_file)

        # 2. Find DICOM folder (handles nested zips)
        dicom_root = find_dicom_folder(extract_dir)

        # 3. Pick the series with most slices
        dicom_files = get_series_with_max_slices(dicom_root)

        # 4. Load HU-volume for segmentation
        hu_volume = load_hu_volume(dicom_files)

        # 5. Read full series into ITK image for geometry reference
        sitk_reader = sitk.ImageSeriesReader()
        sitk_reader.SetFileNames(dicom_files)
        series_itk = sitk_reader.Execute()

        # 6. Save unregistered HU-NIfTI (correct dimensions & spacing)
        os.makedirs("nifti_output", exist_ok=True)
        unreg_nifti = "nifti_output/unregistered.nii.gz"
        unreg_img = sitk.GetImageFromArray(hu_volume)
        unreg_img.CopyInformation(series_itk)
        sitk.WriteImage(unreg_img, unreg_nifti)
        logger.info(f"Saved unregistered HU NIfTI: {unreg_nifti}")

        # 7. ANTs registration
        atlas = "MNI152_T1_1mm.nii.gz"
        registered_nifti = run_ants_registration(unreg_nifti, atlas, "nifti_output")

        # 8. Midline‑shift pipeline
        vol_np = sitk.GetArrayFromImage(sitk.ReadImage(registered_nifti))
        ref_img = sitk.ReadImage(registered_nifti)
        HU_MIN, HU_MAX = -5, 75
        out_dir = "final_output"
        os.makedirs(out_dir, exist_ok=True)

        masks = segment_volume_threshold(vol_np)

        idx = max(range(masks.shape[0]), key=lambda i: np.count_nonzero(masks[i]))
        logger.info(f"Ideal slice = {idx}")

        _, mid_ideal, _ = compute_ideal_midline_on_slice(
            ct_np=vol_np,
            ideal_slice_index=idx,
            output_dir=out_dir,
            hu_min=HU_MIN,
            hu_max=HU_MAX
        )
        actual_mask = np.zeros_like(masks, dtype=np.uint8)
        actual_mask[idx] = estimate_actual_midline_mask_on_slice(masks[idx])

        # Save masks as NIfTI
        ideal_path = os.path.join(out_dir, "ideal.nii.gz")
        actual_path = os.path.join(out_dir, "actual.nii.gz")
        for arr, path in [(mid_ideal, ideal_path), (actual_mask, actual_path)]:
            img = sitk.GetImageFromArray(arr)
            img.CopyInformation(ref_img)
            sitk.WriteImage(img, path)
            logger.info(f"Saved mask: {path}")

        # Compute and return shift
        _, avg_shift = calculate_midline_shift_mm(
            ideal_midline_path=ideal_path,
            actual_midline_path=actual_path,
            output_path=os.path.join(out_dir, "shifts.npz")
        )
        logger.info(f"Midline shift = {avg_shift:.2f} mm")

        return {"ideal_slice_index": idx, "midline_shift_mm": float(avg_shift)}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error in /predict_from_dicom:\n" + tb)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})
