"""Util methods for server."""

import os
import base64
import json
import joblib
import cv2
import numpy as np
from wavelet import (
    w2d,
)  # Importing the wavelet transformation function from a custom module

# Global variables to store mappings and model
__class_name_to_number = {}  # Dictionary to map class names to numbers
__class_number_to_name = {}  # Dictionary to map class numbers to names
__model = None  # Global variable to store the trained machine learning model


def classify_image(image_base64_data, file_path=None):
    # Function to classify the input image using the loaded model
    imgs = get_cropped_image_if_2_eyes(
        file_path, image_base64_data
    )  # Get cropped images with two eyes
    result = []  # List to store classification results for each image

    for img in imgs:
        # Resize the image to a fixed size (32x32) for consistency
        scalled_raw_img = cv2.resize(img, (32, 32))

        # Apply wavelet transformation (w2d function) on the image
        img_har = w2d(img, "db1", 5)

        # Resize the wavelet transformed image to the same size (32x32)
        scalled_img_har = cv2.resize(img_har, (32, 32))

        # Stack the raw image and wavelet transformed image vertically
        combined_img = np.vstack(
            (
                scalled_raw_img.reshape(
                    32 * 32 * 3, 1
                ),  # Flatten and reshape raw image
                scalled_img_har.reshape(
                    32 * 32, 1
                ),  # Flatten and reshape transformed image
            )
        )

        # Calculate the length of the combined image array
        len_image_array = 32 * 32 * 3 + 32 * 32

        # Reshape the combined image array as a single-row array of floats
        final = combined_img.reshape(1, len_image_array).astype(float)

        # Perform classification using the loaded model on the final image array
        result.append(
            {
                "class": class_number_to_name(
                    __model.predict(final)[0]
                ),  # Predicted class name
                "class_probability": np.round(
                    __model.predict_proba(final) * 100, 2
                ).tolist()[
                    0
                ],  # Predicted class probabilities
                "class_dictionary": __class_name_to_number,  # Class name to number mapping
            }
        )

    return result


def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    # Get the directory path where the current script is located
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # load column names from json file using absoulte file path
    class_name_file_path = os.path.join(dir_path, "artifacts", "class_dictionary.json")
    with open(class_name_file_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    # Load the trained model from pickle file using absolute file path
    if __model is None:
        model_file_path = os.path.join(dir_path, "artifacts", "saved_model.pkl")
        with open(model_file_path, "rb") as f:
            __model = joblib.load(f)

    print("Loading saved artifacts...done")


def class_number_to_name(class_number: int) -> str:
    return __class_number_to_name[class_number]


def get_cv2_image_from_base64_string(b64str):
    """
    Decode a base64-encoded image and convert it to a CV2 (OpenCV) image object.

    :param b64str: Base64-encoded image data string.
    :return: CV2 image object.
    """
    # Extract the base64-encoded image data from the string (after the comma)
    encoded_data = b64str.split(",")[1]

    # Decode the base64 data and convert it into a NumPy array of uint8 (unsigned integer) type
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)

    # Decode the NumPy array into a CV2 image object (IMREAD_COLOR for color images)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    """
    Extract cropped faces from an image if two eyes are detected.

    :param image_path: Path to the image file (if available).
    :param image_base64_data: Base64-encoded image data (if image_path is not provided).
    :return: List of cropped face images (CV2 image objects).
    """
    # Load Haar cascade classifiers for face and eye detection
    dir_path = os.path.dirname(os.path.realpath(__file__))
    face_cascade_path = os.path.join(
        dir_path, "opencv", "haarcascades", "haarcascade_frontalface_default.xml"
    )
    eye_cascade_path = os.path.join(
        dir_path, "opencv", "haarcascades", "haarcascade_eye.xml"
    )
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # Read the image based on whether image_path is provided or use base64 data
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    # Loop over detected faces and extract cropped images if two eyes are detected
    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]  # Region of interest in grayscale
        roi_color = img[y : y + h, x : x + w]  # Region of interest in color
        eyes = eye_cascade.detectMultiScale(roi_gray)  # Detect eyes within the face
        if len(eyes) >= 2:  # If at least two eyes are detected
            cropped_faces.append(roi_color)  # Append the cropped face to the list

    return cropped_faces


def get_64_img():
    """
    Load a base64-encoded image from a text file and return the content.

    :return: Base64-encoded image data string.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "b64.txt")
    with open(img_path) as f:
        return f.read()


if __name__ == "__main__":
    # Load saved artifacts (assuming this function is defined elsewhere)
    load_saved_artifacts()

    # Test image classification using an image file path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, "test_images", "sharapova1.jpg")
    print("Image path:", img_path)
    print(
        classify_image(None, img_path)
    )  # Classify the image using the provided image path
