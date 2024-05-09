from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify(
        {
            "name": "Celebrity Image Classifier",
            "version": "1.0.0",
            "ml_algo": "SVM",
            "url": "/classify-image",
        }
    )


@app.route("/classify-image", methods=["GET", "POST"])
def classify_image():
    if request.method == "GET":
        return jsonify("Please post with image data")
    image_data = request.form["image_data"]
    response = jsonify(util.classify_image(image_data, None))
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000)
