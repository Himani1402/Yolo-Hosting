from flask import Flask, request, jsonify
from detection_pipeline.detector import Detector
import os

app = Flask(__name__)

# Load the model once when the app starts
model_path = os.environ.get("MODEL_PATH", "model/yolov8m.xml")
detector = Detector(model_path=model_path, input_size=(640, 640))

@app.route("/")
def home():
    return "YOLO detection server is running!"

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    
    # Run detection (you need to update detector.run() to accept images)
    results = detector.run(image_file)  # you'll probably need to modify this part
    
    return jsonify({"message": "Detection complete", "results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
