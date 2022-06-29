import base64
import cv2
import numpy as np

def decode(image):
    imgdata = base64.b64decode(image)
    image1 = np.asarray(bytearray(imgdata), dtype="uint8")
    image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
    return image1