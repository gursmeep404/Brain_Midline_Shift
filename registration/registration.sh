#!/bin/bash
# Rigid (rotation + translation) registration using atlas
# Applies transform safely using the atlas as reference

# Inputs
PATIENT_NIFTI="brain_normal_5.nii.gz"
ATLAS_NIFTI="MNI152_T1_1mm.nii.gz"
OUTPUT_DIR="output/"
OUTPUT_NAME="registered_normal_5.nii.gz"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Sanity checks
if [ ! -f "$PATIENT_NIFTI" ]; then
    echo "Error: Patient NIfTI not found: $PATIENT_NIFTI"
    exit 1
fi
if [ ! -f "$ATLAS_NIFTI" ]; then
    echo "Error: Atlas NIfTI not found: $ATLAS_NIFTI"
    exit 1
fi

echo "Registering:"
echo "  CT:    $PATIENT_NIFTI"
echo "  Atlas: $ATLAS_NIFTI"

# Step 1: Estimate rigid transform
antsRegistration \
  --dimensionality 3 \
  --float 0 \
  --output [${OUTPUT_DIR}transform_,${OUTPUT_DIR}dummy.nii.gz] \
  --interpolation Linear \
  --use-histogram-matching 0 \
  --winsorize-image-intensities [0.01,0.99] \
  --initial-moving-transform [${ATLAS_NIFTI},${PATIENT_NIFTI},1] \
  --transform Rigid[0.1] \
  --metric MI[${ATLAS_NIFTI},${PATIENT_NIFTI},1,32,Regular,0.25] \
  --convergence [1000x500x250x100,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox

# Step 2: Apply transform
antsApplyTransforms \
  -d 3 \
  -i ${PATIENT_NIFTI} \
  -r ${ATLAS_NIFTI} \
  -o ${OUTPUT_DIR}${OUTPUT_NAME} \
  -t ${OUTPUT_DIR}transform_0GenericAffine.mat \
  -n Linear

echo "Done. Output saved to: ${OUTPUT_DIR}${OUTPUT_NAME}"
