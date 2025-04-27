import logging as log
from collections import deque
from typing import Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from openvino import runtime as ov
from ultralytics.utils import ops


def letterbox(img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize image and padding for detection. Takes image as input,
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

    Parameters:
        img: image for preprocessing
        new_shape: image size after preprocessing in format [width, height]
    Returns:
        img: image after preprocessing
        ratio: height and width scaling ratio
        padding_size: height and width padding size
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[1::-1]  # current shape [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    ratio = r, r  # width, height ratios

    # Calculate new unpadded dimensions
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2  

    if shape != new_unpad:  # resize
        img = cv2.resize(img, dsize=new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  # add border
    
    return img, ratio, (dw, dh)


def preprocess(image: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess image according to YOLOv8 input requirements.

    Parameters:
        image: image for preprocessing
        input_size: image size after preprocessing in format [width, height]
    Returns:
        img: image after preprocessing
    """
    # Add padding to the image
    image = letterbox(image, new_shape=input_size)[0]
    # Convert to float32
    image = image.astype(np.float32)
    # Normalize to (0, 1)
    image /= 255.0
    # Change data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image


def postprocess(pred_boxes: np.ndarray, input_size: Tuple[int, int], orig_img, 
                min_conf_threshold=0.25, nms_iou_threshold=0.45, max_detections=100) -> sv.Detections:
    """
    YOLOv8 model postprocessing function. Applied non-maximum suppression algorithm to detections 
    and rescale boxes to original image size.

    Parameters:
        pred_boxes: model output prediction boxes
        input_size: image size after preprocessing in format [width, height]
        orig_img: image before preprocessing
        min_conf_threshold: minimal accepted confidence for object filtering
        nms_iou_threshold: minimal overlap score for removing objects duplicates in NMS
        max_detections: maximum detections after NMS
    Returns:
        det: detected boxes in sv.Detections format
    """
    # Non-maximum suppression parameters
    nms_kwargs = {"agnostic": False, "max_det": max_detections}
    # Apply non-maximum suppression
    pred = ops.non_max_suppression(torch.from_numpy(pred_boxes), min_conf_threshold, nms_iou_threshold, nc=80, **nms_kwargs)[0]
    
    # No predictions in the image
    if not len(pred):
        return sv.Detections.empty()

    # Transform boxes to pixel coordinates
    pred[:, :4] = ops.scale_boxes(input_size, pred[:, :4], orig_img.shape).round()
    
    # Convert from torch tensor to numpy array
    pred = np.array(pred)
    
    # Create detections in supervision format
    det = sv.Detections(
        xyxy=pred[:, :4],
        confidence=pred[:, 4],
        class_id=pred[:, 5].astype(int)
    )
    
    # Return all detections (no filtering)
    return det


def get_direction(bbox, frame_width):
    """
    Determine the direction (left, center, right) of an object based on its bounding box
    
    Parameters:
        bbox: bounding box coordinates [x_min, y_min, x_max, y_max]
        frame_width: width of the frame
    Returns:
        direction: string indicating "Left", "Center", or "Right"
    """
    # Calculate center x-coordinate of the bounding box
    center_x = (bbox[0] + bbox[2]) / 2
    
    # Determine direction based on position in frame
    if center_x < frame_width / 3:
        return "Left"
    elif center_x < 2 * frame_width / 3:
        return "Center"
    else:
        return "Right"


def get_object_distance(bbox, frame_height, object_class=None):
    """
    Estimate the distance of an object based on its bounding box size
    
    Parameters:
        bbox: bounding box coordinates [x_min, y_min, x_max, y_max]
        frame_height: height of the frame
        object_class: class of the detected object (optional)
    Returns:
        distance: estimated distance in meters
        is_close: boolean indicating if object is in close proximity
    """
    # Calculate height of the bounding box relative to frame height
    box_height = bbox[3] - bbox[1]
    height_ratio = box_height / frame_height
    
    # Different objects have different real-world sizes, so we adjust based on class
    # Only using classes available in the detection model
    if object_class == "person":  # 0: person
        # Average person is about 1.7m tall
        distance = 1.7 / (height_ratio * 1.5)
    elif object_class in ["car", "truck", "bus"]:  # 2: car, 7: truck, 5: bus
        # Vehicles are larger
        distance = 2.0 / (height_ratio * 1.5)
    elif object_class == "bicycle":  # 1: bicycle
        # Medium-sized transportation
        distance = 1.2 / (height_ratio * 1.5)
    elif object_class == "bottle":  # 39: bottle
        # Small container
        distance = 0.25 / (height_ratio * 1.5)
    elif object_class in ["cup", "wine glass"]:  # 41: cup, 40: wine glass
        # Drinking vessels
        distance = 0.15 / (height_ratio * 1.5)
    elif object_class == "cell phone":  # 67: cell phone
        # Small electronic
        distance = 0.15 / (height_ratio * 1.5)
    elif object_class == "book":  # 73: book
        # Reading material
        distance = 0.3 / (height_ratio * 1.5)
    elif object_class == "laptop":  # 63: laptop
        # Computing device
        distance = 0.35 / (height_ratio * 1.5)
    elif object_class == "backpack":  # 24: backpack
        # Student bag
        distance = 0.5 / (height_ratio * 1.5)
    elif object_class == "chair":  # 56: chair
        # Seating
        distance = 0.8 / (height_ratio * 1.5)
    elif object_class == "dining table":  # 60: dining table (desk)
        # Table/desk
        distance = 1.2 / (height_ratio * 1.5)
    elif object_class == "keyboard":  # 66: keyboard
        # Input device
        distance = 0.4 / (height_ratio * 1.5)
    elif object_class == "mouse":  # 64: mouse
        # Small input device
        distance = 0.1 / (height_ratio * 1.5)
    elif object_class == "tv":  # 62: tv (could be monitor)
        # Display screen
        distance = 0.6 / (height_ratio * 1.5)
    elif object_class == "clock":  # 74: clock
        # Time device
        distance = 0.3 / (height_ratio * 1.5)
    elif object_class == "scissors":  # 76: scissors
        # Cutting tool
        distance = 0.15 / (height_ratio * 1.5)
    elif object_class == "bench":  # 13: bench
        # Long seating
        distance = 1.5 / (height_ratio * 1.5)
    elif object_class in ["fork", "knife", "spoon"]:  # 42: fork, 43: knife, 44: spoon
        # Eating utensils
        distance = 0.2 / (height_ratio * 1.5)
    elif object_class == "bowl":  # 45: bowl
        # Container
        distance = 0.2 / (height_ratio * 1.5)
    elif object_class == "tie":  # 27: tie
        # Clothing accessory
        distance = 0.15 / (height_ratio * 1.5)
    elif object_class == "handbag":  # 26: handbag
        # Carrying accessory
        distance = 0.4 / (height_ratio * 1.5)
    elif object_class == "sports ball":  # 32: sports ball
        # Play item
        distance = 0.2 / (height_ratio * 1.5)
    else:
        # Default formula for other objects
        distance = 1.0 / (height_ratio * 1.5)
    
    # Apply some constraints for reasonable values
    distance = max(0.2, min(distance, 20.0))
    
    # Determine if object is in close proximity (less than 3 meters)
    is_close = distance < 3
    
    return distance, is_close


def draw_text(
    image: np.ndarray, 
    text: str, 
    point: tuple, 
    color: tuple = (255, 255, 255), 
    thickness: int = 2
) -> None:
    """
    Draws text on the image with background.

    Parameters:
        image: image to draw on
        text: text to draw
        point: top left corner of the text
        color: text color
        thickness: thickness of the text
    """
    _, f_width = image.shape[:2]
    font_scale = f_width / 1500

    text_size, _ = cv2.getTextSize(
        text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness
    )

    rect_width = text_size[0] + 20
    rect_height = text_size[1] + 20
    rect_x, rect_y = point

    # Draw background rectangle (black color)
    cv2.rectangle(
        image, 
        pt1=(rect_x, rect_y), 
        pt2=(rect_x + rect_width, rect_y + rect_height), 
        color=(0, 0, 0), 
        thickness=cv2.FILLED
    )

    # Calculate centered text position
    text_x = rect_x + (rect_width - text_size[0]) // 2
    text_y = rect_y + (rect_height + text_size[1]) // 2

    # Draw the text
    cv2.putText(
        image, 
        text=text, 
        org=(text_x, text_y), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=font_scale, 
        color=color, 
        thickness=thickness, 
        lineType=cv2.LINE_AA
    )
