import cv2
import pywt
import numpy as np


def w2d(img, mode="haar", level=1):
    # Convert input image to an array (assuming the image is in RGB format)
    im_array = img

    # Convert the RGB image to grayscale
    im_array = cv2.cvtColor(im_array, cv2.COLOR_RGB2GRAY)

    # Convert the grayscale image to float datatype
    im_array = np.float32(im_array)

    # Perform wavelet decomposition on the image array
    coeffs = pywt.wavedec2(im_array, mode, level=level)

    # Process the coefficients
    coeffs_H = list(coeffs)  # Convert coefficients tuple to a list
    coeffs_H[
        0
    ] *= 255  # Amplify the approximation coefficients (low-frequency component)

    # Perform wavelet reconstruction using the processed coefficients
    im_array_H = pywt.waverec2(coeffs_H, mode)

    # Scale the reconstructed image array to the uint8 datatype (0-255 range)
    im_array_H *= 255
    im_array_H = np.uint8(im_array_H)

    return im_array_H
