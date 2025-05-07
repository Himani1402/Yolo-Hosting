from fastapi import FastAPI
from detection_pipeline.detector import Detector

# Initialize FastAPI app
app = FastAPI()

# Load the model once during startup
model_path = "model/yolov8m.xml"  # On Hugging Face, paths should be relative to the repo files
detector = Detector(model_path=model_path, input_size=(640, 640))

@app.get("/run-detection")
def run_detection():
    """
    Endpoint to trigger the detector's run function.
    """
    result = detector.run()
    return {"status": "Detection run completed", "result": result}

