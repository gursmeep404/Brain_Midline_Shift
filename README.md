# Midline Shift Detection and Measurement in Brain CT Scans

Midline Shift is a condition which is always secondary to some other pathology which causes **Intracranial Pressure** inside the skull and displaces normal brain structures. These pathologies usually are `Haemorrhage`, `Tumor`, `Edema`. These take up space inside the skull and hence displace midline structures which include the falx ceribri, septum pellucidum, third ventricle, fourth ventricle, pineal gland and foreman of monro. This is called the midline shift.

![Brain Anatomy](./images/brain_anatomy.png)











---- Approaches and Dead ends

Problem one : Ideal midline
-> solved using 2 things - 
1. Created a bone mask. Centre of mass by symmetry is the ideal midline (fallback)
2. Edge detection using canny and then hough transform to estimate a midline

--- As fas as I know ideal midline is correct

Problem two : Actual midline
-> Two subparts
1. Segmentation of ventricles (midline structures to be precise)
2. Midline through the septum pellucidum

Approaches for segmentation:
1. Thresholding (morphological cleaning. skipping 20% slices in the beginning and end and also trimming 1/3rd mask from up and down to remove noise)
2. Using a segmentation model (SynthSeg)

Approaches for actual midline:
1. Templating
2. PCA


Issues I need to resolve:
- Look for better ways to segment
- Look for better ways to calculate actual midline if PCA doesn't work
- Maybe somehow try to find one ideal slice to perform all the calculations just like how a radiologist does