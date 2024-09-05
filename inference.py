"""
This script performs object detection on images using two YOLO models: one for detecting persons and another for detecting objects
on detected persons.

Usage:
    python script_name.py <person_det_model_path> <object_det_model_path> <input_img_path> <output_path>

Arguments:
    person_det_model_path (str): Path to the YOLO model weights for detecting persons.
    object_det_model_path (str): Path to the YOLO model weights for detecting objects on detected persons.
    input_img_path (str): Path to the input image file.
    output_path (str): Directory where the resulting image with annotations will be saved.

Functions:
    detect_person(person_det_model_path, input_img_path):
        Detects persons in the input image using the specified YOLO model. Crops detected persons and returns the cropped images along with their
        bounding boxes.

    detect_object(cropped_persons, original_image, object_det_model_path):
        Detects objects on cropped images of persons using the specified YOLO model. Maps object bounding boxes back to the original image and 
        draws these bounding boxes.

    save_image(inferred_image, input_img_path, output_img_dir):
        Saves the annotated image to the specified output directory.

    main():
        Parses command-line arguments and initiates the detection and saving process.
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import argparse
import os

def detect_person(person_det_model_path, input_img_path):
    
    """
    Detects persons in the input image using the trained YOLO model. Crops detected persons and returns the cropped images along with their
    bounding boxes.

    Args:
        person_det_model_path (str): Path to the YOLO model weights for detecting persons.
        input_img_path (str): Path to the input image file.

    Returns:
        cropped_persons (list of tuples): List containing cropped images of detected persons and their bounding boxes.
        image (numpy.ndarray): The original image with detected persons.
    """
    
    # Load the YOLO model for person detection
    person_model = YOLO(person_det_model_path)
    
    # Perform person detections on the input image
    person_results = person_model(input_img_path, conf=0.25, iou=0.4)
    
    # Load the image using OpenCV
    image = cv2.imread(input_img_path)
    
    cropped_persons = []
    person_bboxes = []
    
    # Loop over detected persons
    for person_result in person_results:
        # Extract the bounding boxes of the person
        person_bboxes = person_result.boxes.xyxy.cpu().numpy()

        for person_bbox in person_bboxes:
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = [int(coord) for coord in person_bbox]

            # Crop the image of each detected person
            cropped_person = image[y1:y2, x1:x2]
            cropped_persons.append((cropped_person, (x1, y1, x2, y2)))  # Store the cropped person and its original bbox
            
    return cropped_persons, image

def detect_object(cropped_persons, original_image, object_det_model_path):
    
    """
    Detects objects on cropped images of persons using the trained YOLO model. Maps object bounding boxes back to the original image and draws
    these bounding boxes.

    Args:
        cropped_persons (list of tuples): List containing cropped images of detected persons and their bounding boxes.
        original_image (numpy.ndarray): The original image with detected persons.
        object_det_model_path (str): Path to the YOLO model weights for detecting objects.

    Returns:
        original_image (numpy.ndarray): The original image with bounding boxes drawn around detected objects.
    """
    
    # Load the YOLO model for object detection
    object_model = YOLO(object_det_model_path)
    
    # Define labels for the detected objects
    label_names = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]
    
    for cropped_person, person_bbox in cropped_persons:
        x1, y1, x2, y2 = person_bbox  # Original coordinates of the person

        # Detect objects on the cropped image using the second model
        cropped_image_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)  # Convert to RGB for YOLO model
        object_results = object_model(cropped_image_rgb, conf=0.25, iou=0.4)

        for object_result in object_results:
            # Extract bounding boxes and class ids for each object
            object_bboxes = object_result.boxes.xyxy.cpu().numpy()
            class_ids = object_result.boxes.cls.cpu().numpy()

            # Map object bounding boxes back to the original image
            for object_bbox in object_bboxes:
                ox1, oy1, ox2, oy2 = [int(coord) for coord in object_bbox]
                
                # Translate object bbox to original image coordinates
                original_ox1 = x1 + ox1
                original_oy1 = y1 + oy1
                original_ox2 = x1 + ox2
                original_oy2 = y1 + oy2

                # Draw the object bounding box on the original image
                cv2.rectangle(original_image, (original_ox1, original_oy1), (original_ox2, original_oy2), color=(0, 255, 0), thickness=3)

                # Add labels
                label = label_names[int(class_ids)-1]
                cv2.putText(original_image, label, (original_ox1, original_oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return original_image

def save_image(inferred_image, input_img_path, output_img_dir):
    
    """
    Saves the annotated image to the specified output directory.

    Args:
        inferred_image (numpy.ndarray): The image with annotations.
        input_img_path (str): Path to the input image file.
        output_img_dir (str): Directory where the resulting image will be saved.
    """
    
    # Convert BGR image to RGB for saving
    image_rgb = cv2.cvtColor(inferred_image, cv2.COLOR_BGR2RGB)
    
    # Extract file name from input image path. Use this file name to maintain consistency
    filename = os.path.basename(input_img_path)
    
    # Create output directory if it does not exist
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Define the path to save the annoted image
    output_img_path = os.path.join(output_img_dir, filename)
    
    # Save the image using PIL
    pil_image = Image.fromarray(image_rgb)
    pil_image.save(output_img_path)

def main():
    
    """
    Parses command-line arguments and initiates the detection and saving process.
    
    Returns:
        Paths to the person detection model, object detection model, input image, and output directory.
    """
    
    parser = argparse.ArgumentParser(description="Object Detection.")
    parser.add_argument("person_det_model_path", type=str, help="Path to the person detection model weights.")
    parser.add_argument("object_det_model_path", type=str, help="Path to the object detection model weights.")
    parser.add_argument("input_img_path", type=str, help="Input image path")
    parser.add_argument("output_path", type=str, help="Output directory where the inferred image will be stored.")
    args = parser.parse_args()
    
    person_det_model_path = args.person_det_model_path
    object_det_model_path = args.object_det_model_path
    input_img_path = args.input_img_path
    output_img_dir = args.output_path
    
    return person_det_model_path, object_det_model_path, input_img_path, output_img_dir

if __name__ == "__main__":
    # Retrieve paths and image file locations from command-line arguments
    person_det_model_path, object_det_model_path, input_img_path, output_img_dir = main()
    
    # Detect persons in the input image
    cropped_persons, image = detect_person(person_det_model_path, input_img_path)
    
    # Detect objects on the cropped persons and annotate the original image
    inferred_image = detect_object(cropped_persons, image, object_det_model_path)
    
    # Save the annotated image to the specified directory
    save_image(inferred_image, input_img_path, output_img_dir)