import os
import xml.etree.ElementTree as ET
import re
import argparse
from PIL import Image

def convert_voc_to_yolo(voc_path, yolo_ppe_path, yolo_person_path, classes, img_dir, cropped_img_dir):
    """
    Converts Pascal VOC annotations to YOLO format and saves the results to the specified directories.
    Crops persons from the images and adjusts object bounding boxes in the cropped image.

    Args:
        voc_path (str): Directory path containing VOC XML annotations.
        yolo_ppe_path (str): Directory path to save YOLO format annotations for 'ppe objects'.
        yolo_person_path (str): Directory path to save YOLO format annotations for 'person'.
        classes (list of str): List of class names corresponding to their indices.
        img_dir (str): Directory containing the original images.
        cropped_img_dir (str): Directory to save the cropped images of persons.

    The function processes each XML file in the VOC directory, crops the persons, converts the bounding boxes to YOLO format,
    and saves the cropped images and annotations for objects detected on the persons.
    """
    # Create output directories if they don't exist
    os.makedirs(yolo_ppe_path, exist_ok=True)
    os.makedirs(yolo_person_path, exist_ok=True)
    os.makedirs(cropped_img_dir, exist_ok=True)

    # Iterate through each XML file in the input directory
    for xml_file in os.listdir(voc_path):
        if not xml_file.endswith('.xml'):
            continue

        # Parse the XML file
        tree = ET.parse(os.path.join(voc_path, xml_file))
        root = tree.getroot()

        # Get the image dimensions and load the image
        image_filename = root.find('filename').text
        image_path = os.path.join(img_dir, image_filename)
        image = Image.open(image_path)
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)

        person_bboxes = []

        for obj in root.iter('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue

            class_id = classes.index(class_name)

            # Get bounding box coordinates from the XML
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # If the class is 'person', store the bounding box to crop the person later
            if class_id == 0:
                person_bboxes.append((xmin, ymin, xmax, ymax))

        # Crop each person and adjust object bounding boxes within the person's cropped area
        for i, (xmin, ymin, xmax, ymax) in enumerate(person_bboxes):
            # Crop the person from the original image
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            cropped_image_filename = f"{xml_file.replace('.xml', '')}_person_{i}.jpg"
            cropped_image_path = os.path.join(cropped_img_dir, cropped_image_filename)
            cropped_image.save(cropped_image_path)

            # Open YOLO annotation files for person and objects
            with open(os.path.join(yolo_ppe_path, f"{cropped_image_filename.replace('.jpg', '.txt')}"), 'w') as yolo_ppe_file:
                with open(os.path.join(yolo_person_path, f"{cropped_image_filename.replace('.jpg', '.txt')}"), 'w') as yolo_person_file:
                    # Write the person's YOLO bbox
                    yolo_person_file.write(f"0 0.5 0.5 1.0 1.0\n")

                    # Now adjust object bboxes relative to the cropped person
                    for obj in root.iter('object'):
                        class_name = obj.find('name').text
                        if class_name not in classes:
                            continue
                        class_id = classes.index(class_name)

                        # Skip the 'person' class as it's already handled
                        if class_id == 0:
                            continue

                        # Get object bounding box and check if it lies within the person bbox
                        bndbox = obj.find('bndbox')
                        obj_xmin = int(bndbox.find('xmin').text)
                        obj_ymin = int(bndbox.find('ymin').text)
                        obj_xmax = int(bndbox.find('xmax').text)
                        obj_ymax = int(bndbox.find('ymax').text)

                        # If the object's bbox lies within the person bbox, adjust it for the cropped image
                        if obj_xmin >= xmin and obj_ymin >= ymin and obj_xmax <= xmax and obj_ymax <= ymax:
                            cropped_xmin = obj_xmin - xmin
                            cropped_ymin = obj_ymin - ymin
                            cropped_xmax = obj_xmax - xmin
                            cropped_ymax = obj_ymax - ymin

                            # Convert to YOLO format for the cropped image
                            crop_width = xmax - xmin
                            crop_height = ymax - ymin
                            x_center = (cropped_xmin + cropped_xmax) / 2.0 / crop_width
                            y_center = (cropped_ymin + cropped_ymax) / 2.0 / crop_height
                            bbox_width = (cropped_xmax - cropped_xmin) / crop_width
                            bbox_height = (cropped_ymax - cropped_ymin) / crop_height

                            yolo_ppe_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")


# Run the conversion
def main():
    """
    Parses command-line arguments and runs the conversion process.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotations to YOLO format and crop persons from the images.")
    parser.add_argument("input_dir", type=str, help="Base input directory path containing VOC annotations.")
    parser.add_argument("output_dir_ppe", type=str, help="Output directory where YOLO annotations will be saved.")
    parser.add_argument("output_dir_person", type=str, help="Output directory where YOLO annotations will be saved.")
    parser.add_argument("class_path", type=str, help="Path to the text file containing classes.")
    parser.add_argument("img_dir", type=str, help="Directory containing the original images.")
    parser.add_argument("cropped_img_dir", type=str, help="Directory to save the cropped images.")
    args = parser.parse_args()

    # Use the parsed arguments
    input_dir = args.input_dir
    output_dir_ppe = args.output_dir_ppe
    output_dir_person = args.output_dir_person
    class_path = args.class_path
    img_dir = args.img_dir
    cropped_img_dir = args.cropped_img_dir

    classes = []
    
    # Get the class names from the class file
    with open(class_path, 'r') as file:
        classes = file.read().strip().splitlines()

    # Run the conversion
    convert_voc_to_yolo(input_dir, output_dir_ppe, output_dir_person, classes, img_dir, cropped_img_dir)


if __name__ == "__main__":
    main()