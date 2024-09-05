import cv2
import os

# Define paths
image_path = 'input_data/images/-184-_png_jpg.rf.b02963998a79b9ad5079f57b65130bc2.jpg'
txt_path = 'output_data/ppe/-184-_png_jpg.rf.b02963998a79b9ad5079f57b65130bc2.txt'

# Load the image
image = cv2.imread(image_path)
height, width, _ = image.shape

# Read YOLO format annotations
with open(txt_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        bbox_width = float(parts[3]) * width
        bbox_height = float(parts[4]) * height
        
        # Convert to pixel coordinates
        xmin = int(x_center - bbox_width / 2)
        ymin = int(y_center - bbox_height / 2)
        xmax = int(x_center + bbox_width / 2)
        ymax = int(y_center + bbox_height / 2)
        
        # Draw bounding box and label
        color = (0, 255, 0)  # Green color for bounding box
        thickness = 2  # Thickness of the bounding box
        label = f"Class {class_id}"
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        image = cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)


cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(4000)
cv2.destroyAllWindows()
