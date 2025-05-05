from fastapi import FastAPI
from pydantic import BaseModel
from detection_pipeline.detector import Detector

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for the input data (adjust fields based on your need)
class DetectionRequest(BaseModel):
    model_path: str
    input_size: tuple

# Define a POST endpoint to trigger the detection
@app.post("/run-detection")
async def run_detection(request: DetectionRequest):
    try:
        # Instantiate the Detector with the model path and input size
        detector = Detector(model_path=request.model_path, input_size=request.input_size)
        
        # Run the detector (assuming this doesn't block the main thread too much)
        detector.run()
        
        return {"message": "Detection run successfully", "status": "success"}
    except Exception as e:
        return {"message": str(e), "status": "error"}

# Main function to test locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
