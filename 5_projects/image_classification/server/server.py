from flask import Flask, request, jsonify
import util  # Importing custom utility module named 'util' for image classification

app = Flask(__name__)  # Create a Flask web application instance


# @app.route("/")
# def home():
#     return jsonify(
#         {
#             "name": "Celebrity Image Classifier",
#             "version": "1.0.0",
#             "ml_algo": "SVM",  # Machine learning algorithm used (Support Vector Machine)
#             "url": "/classify-image",  # Endpoint URL for image classification
#         }
#     )


@app.route("/classify-image", methods=["GET", "POST"])
def classify_image():
    """
    Route handler for the "/classify-image" endpoint.
    Handles GET and POST requests to classify images.
    """
    if request.method == "GET":
        return jsonify(
            "Please post with image data"
        )  # Respond with a message for GET requests

    # Extract image data from the request form data
    image_data = request.form["image_data"]

    # Call the utility function 'util.classify_image' to classify the image
    response = jsonify(util.classify_image(image_data, None))

    # Add CORS headers to allow cross-origin requests from any domain
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response  # Return the classification result as a JSON response


if __name__ == "__main__":
    # Start the Flask web server on port 5000
    print("Starting Python Flask Server For Sports Celebrity Image Classification")

    # Load any saved ML model artifacts required for image classification
    util.load_saved_artifacts()

    # Run the Flask application on port 5000
    app.run(port=5000)
