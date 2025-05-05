from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Dummy detector class for demonstration (replace with your actual detector)
class DummyDetector:
    def run(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")
        # Implement video processing logic here
        return {"result": "dummy detection on video " + video_path}

# Initialize detector
detector = DummyDetector()

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
        
        # Log the incoming file name (for debugging)
        print("Received file:", file.filename)

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Step 2: Check if the file is a valid video
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed types are: mp4, avi, mov, mkv."}), 400

        # Step 3: Save the uploaded file
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)

        # Step 4: Run detection
        results = detector.run(filepath)

        # Step 5: Return results
        return jsonify({"message": "Detection completed", "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
