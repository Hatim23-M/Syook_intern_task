from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import argparse
import os

def detect_person(person_det_model_path, input_img_path):
    
    person_model = YOLO(person_det_model_path)
    person_results = person_model(input_img_path, conf=0.25, iou=0.4)
    
    # Load the image using OpenCV
    image = cv2.imread(input_img_path)
    cropped_persons = []
    person_bboxes = []
    
    # Loop over the detected persons
    for person_result in person_results:
        person_bboxes = person_result.boxes.xyxy.cpu().numpy()  # Bounding boxes of persons

        for person_bbox in person_bboxes:
            x1, y1, x2, y2 = [int(coord) for coord in person_bbox]

            # Step 2: Crop the image of each detected person
            cropped_person = image[y1:y2, x1:x2]
            cropped_persons.append((cropped_person, (x1, y1, x2, y2)))  # Store the cropped person and its original bbox
            
    return cropped_persons, image

def detect_object(cropped_persons, original_image, object_det_model_path):
    object_model = YOLO(object_det_model_path)
    label_names = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]
    
    for cropped_person, person_bbox in cropped_persons:
        x1, y1, x2, y2 = person_bbox  # Original coordinates of the person

        # Step 3: Detect objects on the cropped image using the second model
        cropped_image_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)  # Convert to RGB for YOLO model
        object_results = object_model(cropped_image_rgb, conf=0.25, iou=0.4)

        for object_result in object_results:
            object_bboxes = object_result.boxes.xyxy.cpu().numpy()  # Detected objects
            class_ids = object_result.boxes.cls.cpu().numpy()       # Extract class IDs

            # Step 4: Map object bounding boxes back to the original image
            for object_bbox in object_bboxes:
                ox1, oy1, ox2, oy2 = [int(coord) for coord in object_bbox]
                
                # Translate object bbox to original image coordinates
                original_ox1 = x1 + ox1
                original_oy1 = y1 + oy1
                original_ox2 = x1 + ox2
                original_oy2 = y1 + oy2

                # Draw the object bounding box on the original image
                cv2.rectangle(original_image, (original_ox1, original_oy1), (original_ox2, original_oy2), color=(0, 255, 0), thickness=3)

                # Optionally add labels (you can modify this part as needed)
                label = label_names[int(class_ids)-1]
                cv2.putText(original_image, label, (original_ox1, original_oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return original_image

def save_image(inferred_image, input_img_path, output_img_dir):
    
    image_rgb = cv2.cvtColor(inferred_image, cv2.COLOR_BGR2RGB)
    
    filename = os.path.basename(input_img_path)
    os.makedirs(output_img_dir, exist_ok=True)
    output_img_path = os.path.join(output_img_dir, filename)
    
    pil_image = Image.fromarray(image_rgb)
    pil_image.save(output_img_path)

def main():
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
    
    person_det_model_path, object_det_model_path, input_img_path, output_img_dir = main()
    
    cropped_persons, image = detect_person(person_det_model_path, input_img_path)
    inferred_image = detect_object(cropped_persons, image, object_det_model_path)
    
    save_image(inferred_image, input_img_path, output_img_dir)