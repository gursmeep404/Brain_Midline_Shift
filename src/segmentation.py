def segment_ventricles(slice_img):
    # Invert and threshold to get dark CSF/ventricles
    blurred = cv2.GaussianBlur(slice_img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return morph