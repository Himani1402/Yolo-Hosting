from detection_pipeline.detector import Detector

def main():
    # Path to the OpenVINO IR model
    model_path = "model/yolov8m_openvino_int8_model/yolov8m.xml"

    detector = Detector(model_path=model_path, input_size=(640, 640))
    detector.run()

if __name__ == "__main__":
    main()
