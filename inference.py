import os
import argparse
from ultralytics import YOLO
import cv2
import numpy as np

# Hardcoded class names
PERSON_CLASS_NAMES = ['person']
PPE_CLASS_NAMES = ['hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']

def load_model(model_path):
    return YOLO(model_path)

def detect_objects(model, image, conf_threshold=0.3):
    results = model(image, conf=conf_threshold)
    return results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy(), results[0].boxes.conf.cpu().numpy()

def draw_boxes(image, boxes, classes, scores, class_names, color, is_ppe=False):
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box[:4])
        label = f'{class_names[int(cls)]} {score:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)  # White color
        
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        if is_ppe:
            text_offset_x = x1
            text_offset_y = y2 + text_height + 5
        else:
            text_offset_x = x1
            text_offset_y = y1 - 5
        
        cv2.rectangle(image, (text_offset_x, text_offset_y - text_height - 5),
                      (text_offset_x + text_width, text_offset_y + 5), color, -1)
        
        cv2.putText(image, label, (text_offset_x, text_offset_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    
    return image

def process_image(image_path, person_model, ppe_model, person_conf, ppe_conf):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None

    person_boxes, person_classes, person_scores = detect_objects(person_model, image, person_conf)
    image = draw_boxes(image, person_boxes, person_classes, person_scores, PERSON_CLASS_NAMES, color=(0, 255, 0))

    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        person_crop = image[y1:y2, x1:x2]
        
        ppe_boxes, ppe_classes, ppe_scores = detect_objects(ppe_model, person_crop, ppe_conf)
        
        for ppe_box, ppe_class, ppe_score in zip(ppe_boxes, ppe_classes, ppe_scores):
            ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_box
            orig_x1 = int(x1 + ppe_x1)
            orig_y1 = int(y1 + ppe_y1)
            orig_x2 = int(x1 + ppe_x2)
            orig_y2 = int(y1 + ppe_y2)
            
            image = draw_boxes(image, 
                               [[orig_x1, orig_y1, orig_x2, orig_y2]], 
                               [ppe_class], 
                               [ppe_score], 
                               PPE_CLASS_NAMES, 
                               color=(255, 0, 0), 
                               is_ppe=True)

    return image

def main():
    parser = argparse.ArgumentParser(description="Detect persons and PPE in images.")
    parser.add_argument("input_dir", help="Path to the directory containing input images")
    parser.add_argument("output_dir", help="Path to the directory to save annotated images")
    parser.add_argument("person_model_path", help="Path to the trained person detection model")
    parser.add_argument("ppe_model_path", help="Path to the trained PPE detection model")
    parser.add_argument("--person_conf", type=float, default=0.3, help="Confidence threshold for person detection")
    parser.add_argument("--ppe_conf", type=float, default=0.3, help="Confidence threshold for PPE detection")
    
    args = parser.parse_args()

    person_model = load_model(args.person_model_path)
    ppe_model = load_model(args.ppe_model_path)

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, f"annotated_{filename}")

            annotated_image = process_image(input_path, person_model, ppe_model, 
                                            args.person_conf, args.ppe_conf)

            if annotated_image is not None:
                cv2.imwrite(output_path, annotated_image)
                print(f"Processed: {filename}")
            else:
                print(f"Failed to process: {filename}")

if __name__ == "__main__":
    main()