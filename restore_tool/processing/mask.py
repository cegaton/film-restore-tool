import numpy as np
import cv2

def auto_detect_mask(img):
    gray = np.mean(img, axis=2)
    g = gray / (np.max(gray) + 1e-6)

    mask = g > 0.02

    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    )

    return mask.astype(np.float32)
