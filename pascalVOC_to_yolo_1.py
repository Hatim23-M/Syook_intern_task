"""
This script converts Pascal VOC annotations to YOLO format for the given object detection  task.

The Pascal VOC format includes XML files that contains bounding box annotations. These annotations are converted into the YOLO format text files. 
YOLO format represents bounding boxes as normalized coordinates stored in a txt file.

Usage:
    python <script_path> <input_dir>  <output_dir_person> <class_path>

Arguments:
    input_dir (str): The base directory path containing Pascal VOC XML annotations.
    output_dir_person (str): The directory where YOLO annotations for 'person' will be saved.
    class_path (str): The path to the text file containing class names, one per line.
    
(There is only one output directory. Here the bbox data of persons for each image is stored.)

Functions:
    convert_voc_to_yolo(voc_path, yolo_ppe_path, yolo_person_path, classes):
        Converts VOC annotations to YOLO format and saves the results to the specified directories.

    main():
        Parses command-line arguments and runs the conversion process.
"""

import os
import xml.etree.ElementTree as ET
import re
import argparse

def convert_voc_to_yolo(voc_path, yolo_person_path, classes):
    """
    Converts Pascal VOC annotations to YOLO format and saves the results to the specified directories.

    Args:
        voc_path (str): Directory path containing VOC XML annotations.
        yolo_person_path (str): Directory path to save YOLO format annotations for 'person'.
        classes (list of str): List of class names corresponding to their indices.

    The function processes each XML file in the VOC directory, converts the bounding boxes to YOLO format,
    and saves the converted annotations to separate files based on the class type.
    """
    # Checks whether the specified path exists or not. If not then it creates those output directories.
    
    if not os.path.exists(yolo_person_path):
        os.makedirs(yolo_person_path)
    
    # Iterates through each file in the input directory
    for xml_file in os.listdir(voc_path):
        if not xml_file.endswith('.xml'):
            continue

        # Parses the XML file
        tree = ET.parse(os.path.join(voc_path, xml_file))
        root = tree.getroot()
        
        # Gets the image dimensions
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)
        
        # Opens a new txt file with the same name as the XML file. 
        with open(os.path.join(yolo_person_path, xml_file.replace('.xml', '.txt')), 'w') as yolo_person_file:
            
            for obj in root.iter('object'):
                class_name = obj.find('name').text
                if class_name not in classes:
                    continue
                
                class_id = classes.index(class_name)
                
                # Gets the dimensions from the xml file
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # Converts to YOLO format
                x_center = (xmin + xmax) / 2.0 / image_width
                y_center = (ymin + ymax) / 2.0 / image_height
                bbox_width = (xmax - xmin) / image_width
                bbox_height = (ymax - ymin) / image_height
                
                # If the class id is 0 (which indicates person) then store it in the first directory, otherwise the second directory.      
                if class_id == 0:
                    yolo_person_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
                
                
# Run the conversion
def main():
    """
    Parses command-line arguments and runs the conversion process.
    """
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotations to YOLO format.")
    parser.add_argument("input_dir", type=str, help="Base input directory path containing VOC annotations.")
    parser.add_argument("output_dir_person", type=str, help="Output directory where YOLO annotations will be saved.")
    parser.add_argument("class_path", type=str, help="Path to the text file containing classes.")
    args = parser.parse_args()

    # Use the parsed arguments
    input_dir = args.input_dir
    output_dir_person = args.output_dir_person
    class_path = args.class_path
    classes = []
    
    # Get the class names from the class file.
    with open(class_path, 'r') as file:
        content = file.read()
    
        split_each_line = re.split(r'\n', content)
    
        for each_class in split_each_line:
            classes.append(each_class)
            
    # Run the conversion
    convert_voc_to_yolo(input_dir, output_dir_person, classes)

if __name__ == "__main__":
    main()