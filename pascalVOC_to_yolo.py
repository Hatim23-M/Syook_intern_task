import os
import xml.etree.ElementTree as ET
import re
import argparse

def convert_voc_to_yolo(voc_path, yolo_path, classes):
    if not os.path.exists(yolo_path):
        os.makedirs(yolo_path)
        
    for xml_file in os.listdir(voc_path):
        if not xml_file.endswith('.xml'):
            continue

        # Parse XML
        tree = ET.parse(os.path.join(voc_path, xml_file))
        root = tree.getroot()
        
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)
        
        with open(os.path.join(yolo_path, xml_file.replace('.xml', '.txt')), 'w') as yolo_file:
            for obj in root.iter('object'):
                class_name = obj.find('name').text
                if class_name not in classes:
                    continue
                
                class_id = classes.index(class_name)
                
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # Convert to YOLO format
                x_center = (xmin + xmax) / 2.0 / image_width
                y_center = (ymin + ymax) / 2.0 / image_height
                bbox_width = (xmax - xmin) / image_width
                bbox_height = (ymax - ymin) / image_height
                
                # Write to file
                yolo_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

# Run the conversion
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotations to YOLO format.")
    parser.add_argument("input_dir", type=str, help="Base input directory path containing VOC annotations.")
    parser.add_argument("output_dir", type=str, help="Output directory where YOLO annotations will be saved.")
    parser.add_argument("class_path", type=str, help="Path to the text file containing classes.")
    args = parser.parse_args()

    # Use the parsed arguments
    input_dir = args.input_dir
    output_dir = args.output_dir
    classes = []
    
    with open('Syook_intern_task/input_data/classes.txt', 'r') as file:
        content = file.read()
    
        split_each_line = re.split(r'\n', content)
    
        for each_class in split_each_line:
            classes.append(each_class)
            
    # Run the conversion
    convert_voc_to_yolo(input_dir, output_dir, classes)

if __name__ == "__main__":
    main()