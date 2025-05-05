from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from detection_pipeline.detector import Detector  # Import your real detector

app = Flask(__name__)

# Initialize the real detector (adjust path to your model .xml)
detector = Detector(model_path="yolo_model/yolov8.xml")  # Replace with correct model path

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return "YOLO detection server is running!"

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Step 1: Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        print("Received file:", file.filename)

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Step 2: Check if file is a valid video
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed types are: mp4, avi, mov, mkv."}), 400

        # Step 3: Save uploaded file
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)

        # Step 4: Process the video and run detection
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({"error": "Failed to read uploaded video."}), 500

        detected_objects = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Run detection on frame
            detections = detector.predict(frame)

            # Extract object names and confidence scores
            for bbox, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                label = detector.class_labels[class_id]
                detected_objects.append({
                    "object": label,
                    "confidence": float(f"{confidence:.2f}")
                })

        cap.release()

        # Get unique objects and count occurrences
        object_summary = {}
        for obj in detected_objects:
            name = obj['object']
            object_summary[name] = object_summary.get(name, 0) + 1

        # Step 5: Return detection summary
        return jsonify({
            "message": "Detection completed",
            "results": {
                "video": filepath,
                "objects_detected": object_summary,
                "total_detections": len(detected_objects)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
