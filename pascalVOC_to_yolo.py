import os
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
import shutil

def copy_image(xml_file, input_dir, output_dir):
    # Construct paths for the given structure
    xml_path = os.path.join(input_dir, 'labels', xml_file)
    image_file = os.path.splitext(xml_file)[0] + '.jpg'  # Assuming JPG format
    image_path = os.path.join(input_dir, 'images', image_file)
    
    # Check for other common image formats if JPG is not found
    if not os.path.exists(image_path):
        for ext in ['.png', '.jpeg', '.JPEG', '.PNG']:
            alt_image_file = os.path.splitext(xml_file)[0] + ext
            alt_image_path = os.path.join(input_dir, 'images', alt_image_file)
            if os.path.exists(alt_image_path):
                image_path = alt_image_path
                image_file = alt_image_file
                break
    
    # Destination path
    dst_path = os.path.join(output_dir, 'images', image_file)
    
    # Copy the image file
    if os.path.exists(image_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(image_path, dst_path)
        return True
    else:
        print(f"Warning: Image file not found for {xml_file}")
        return False

def converter(input_dir, output_dir):
    # Predefined order of classes
    class_names = ['person', 'hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']

    # Create the output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Counter for statistics
    label_counter = Counter()
    unknown_labels = Counter()
    processed_files = 0
    empty_files = 0
    copied_images = 0

    # Converting each annotation to YOLO format
    labels_dir = os.path.join(input_dir, 'labels')
    for annotation_file in os.listdir(labels_dir):
        if annotation_file.endswith('.xml'):
            processed_files += 1
            tree = ET.parse(os.path.join(labels_dir, annotation_file))
            root = tree.getroot()

            image_size = root.find('size')
            image_width = int(image_size.find('width').text)
            image_height = int(image_size.find('height').text)

            yolo_label_file = os.path.join(output_dir, 'labels', os.path.splitext(annotation_file)[0] + '.txt')
            labels_in_file = 0
            
            with open(yolo_label_file, 'w') as output_file:
                for object_tag in root.findall('object'):
                    name_tag = object_tag.find('name')
                    if name_tag is None:
                        print(f"Warning: Missing name tag in file {annotation_file}")
                        continue
                    
                    label = name_tag.text.strip().lower()  # Convert to lowercase and remove whitespace
                    if label not in class_names:
                        unknown_labels[label] += 1
                        print(f"Warning: Unknown class '{label}' in file {annotation_file}")
                        continue
                    
                    class_index = class_names.index(label)
                    label_counter[label] += 1
                    labels_in_file += 1

                    bndbox = object_tag.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymin = float(bndbox.find('ymin').text)
                    ymax = float(bndbox.find('ymax').text)

                    # Ensure bounding box coordinates are within image boundaries
                    xmin = max(0, min(xmin, image_width - 1))
                    xmax = max(0, min(xmax, image_width - 1))
                    ymin = max(0, min(ymin, image_height - 1))
                    ymax = max(0, min(ymax, image_height - 1))

                    # Check if bounding box is valid
                    if xmax <= xmin or ymax <= ymin:
                        print(f"Warning: Invalid bounding box in file {annotation_file}")
                        continue

                    x_center = (xmin + xmax) / 2 / image_width
                    y_center = (ymin + ymax) / 2 / image_height
                    bndbox_width = (xmax - xmin) / image_width
                    bndbox_height = (ymax - ymin) / image_height

                    output_file.write(f"{class_index} {x_center:.6f} {y_center:.6f} {bndbox_width:.6f} {bndbox_height:.6f}\n")
            
            if labels_in_file == 0:
                empty_files += 1
                print(f"Warning: No valid labels found in file {annotation_file}")
            else:
                # Copy the corresponding image
                if copy_image(annotation_file, input_dir, output_dir):
                    copied_images += 1

    print("\nConversion completed. Summary:")
    print(f"Processed files: {processed_files}")
    print(f"Files with no valid labels: {empty_files}")
    print(f"Images copied: {copied_images}")
    print("\nLabel counts:")
    for label, count in label_counter.most_common():
        print(f"{label}: {count}")
    print("\nCleaned and Saved labels in YOLO format and copied images corresponding to the cleaned labels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLO format and copy corresponding images.")
    parser.add_argument("input_dir", help="Directory containing 'labels' (XML files) and 'images' subdirectories")
    parser.add_argument("output_dir", help="Directory to save YOLO format labels and copied images")
    args = parser.parse_args()

    converter(args.input_dir, args.output_dir)