from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Dummy detector class for demonstration (replace with your actual detector)
class DummyDetector:
    def run(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        return {"result": "dummy detection on " + image_path}

# Initialize detector
detector = DummyDetector()

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
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Step 2: Save uploaded file
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)

        # Step 3: Run detection
        # Pass the file path to the detector
        results = detector.run(filepath)

        # Step 4: Return results
        return jsonify({"message": "Detection completed", "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # Update the port to 8080
