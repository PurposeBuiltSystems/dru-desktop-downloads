"""Read camera images through Windows Unicode paths while retaining OpenCV decoding."""
import os
import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    if os.name == 'nt':
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    return cv2.imread(str(path), flags)
