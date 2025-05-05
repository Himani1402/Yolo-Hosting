from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "YOLO detection server is running!"

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Dummy response for now
    return jsonify({'message': 'Detection complete!'}), 200

if __name__ == '__main__':
    app.run()
