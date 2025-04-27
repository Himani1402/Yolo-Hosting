import cv2
import numpy as np
from openvino import runtime as ov
import logging as log
import supervision as sv


from .utils import preprocess, postprocess, get_direction, get_object_distance, draw_text

class Detector:
    def __init__(self, model_path: str, input_size=(640, 640)):
        self.input_size = input_size

        # Initialize OpenVINO Core and Load Model
        self.core = ov.Core()
        self.model = self.core.read_model(model=model_path)
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU")

        # Get model input/output layer
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Setup label map (COCO labels)
        self.class_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

        
    def predict(self, frame: np.ndarray):
        orig_frame = frame.copy()
        input_tensor = preprocess(frame, self.input_size)

        # Inference
        preds = self.compiled_model([input_tensor])[self.output_layer]

        # Postprocess
        detections = postprocess(preds, self.input_size, orig_frame)
        return detections

    def run(self):
        cap = cv2.VideoCapture(0)  # 0 for default webcam

        if not cap.isOpened():
            print("Error: Cannot access camera.")
            return

        box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        processing_times = []
        frame_count = 0

        print("Starting real-time detection. Press 'q' or 'ESC' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1
            f_height, f_width = frame.shape[:2]

            start_time = cv2.getTickCount()

            detections = self.predict(frame)

            labels = []
            detected_objects = []
            close_objects = []

            for bbox, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                x1, y1, x2, y2 = map(int, bbox)
                label = self.class_labels[class_id]

                # Direction and Distance Estimation
                direction = get_direction([x1, y1, x2, y2], f_width)
                distance, is_close = get_object_distance([x1, y1, x2, y2], f_height, label)

                # Format label
                distance_str = f"{distance:.1f}m"
                label_text = f"{label}: {confidence:.2f} ({direction}, {distance_str})"
                labels.append(label_text)

                # Log detected object
                detected_objects.append(f"{label} ({direction}, {distance_str})")

                # Highlight close objects
                if is_close:
                    close_objects.append((label, direction, distance))

                    # Red bounding box for close object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # Draw warning text at top
                    warning_text = f"WARNING: {label} {distance_str} AWAY!"
                    warning_position = (int(f_width / 2 - 150), 50)
                    draw_text(frame, warning_text, warning_position, color=(0, 0, 255), thickness=2)
                
                else:
                    # Normal green bounding box for non-close objects
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Only log unique detections to reduce terminal spam
                if detected_objects:
                    unique_objects = set(detected_objects)
                    log.info(f"Detected: {', '.join(unique_objects)}")


                # Draw direction and distance below the box
                direction_text = f"{direction} - {distance:.1f}m"
                cv2.putText(frame, direction_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Display close object summary at the bottom
            if close_objects:
                close_text = "CLOSE OBJECTS: " + ", ".join([f"{obj[0]} ({obj[1]}, {obj[2]:.1f}m)" for obj in close_objects[:3]])
                close_position = (10, f_height - 50)
                draw_text(frame, close_text, close_position, color=(0, 0, 255), thickness=2)

            # FPS Calculation
            end_time = cv2.getTickCount()
            time_taken = (end_time - start_time) / cv2.getTickFrequency()
            processing_times.append(time_taken)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_time = np.mean(processing_times)
            fps = 1 / avg_time if avg_time > 0 else 0

            # Display FPS
            draw_text(frame, f"Inference: {avg_time*1000:.0f}ms ({fps:.1f} FPS)", (10, 10), color=(255, 255, 0), thickness=2)

            # Final annotated frame (optional if you want box_annotator usage)
            # frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # Show frame
            cv2.imshow("Real-time Object Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Shutting down...")
