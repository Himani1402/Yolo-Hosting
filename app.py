from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
from detection_pipeline.detector import Detector

app = FastAPI()

# Initialize the detector once
model_path = "model\yolov8m_openvino_int8_model\yolov8m.xml"
detector = Detector(model_path=model_path, input_size=(640, 640))

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    # Run detection
    results = detector.run(img)  # Assuming detector.run() can accept an image

    # Return detection results
    return {"detections": results}

if __name__ == "__main__":
    uvicorn.run("your_fastapi_file_name:app", host="0.0.0.0", port=8000, reload=True)
