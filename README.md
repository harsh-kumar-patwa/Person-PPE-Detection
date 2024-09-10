# PPE Detection Project

This project implements a two-stage detection system for Personal Protective Equipment (PPE) in workplace environments. It first detects persons in an image, then identifies PPE items worn by each person.

## Repository Structure

- `weights/`
  - `person_detection_model.pt`: Trained model for person detection
  - `ppe_detection_model.pt`: Trained model for PPE detection
- `inference.py`: Script for running inference on images
- `pascalVOC_to_yolo.py`: Converts Pascal VOC annotations to YOLO format

## Setup

1. Clone this repository or directly download the zip file. :
   ```
   git clone https://github.com/harsh-kumar-patwa/Person-PPE-Detection
   cd Person-PPE-Detection
   ```

2. Install required packages:
   ```
   pip install ultralytics opencv-python
   ```

## Usage

### Converting Annotations

To convert Pascal VOC annotations to YOLO format:

```
python pascalVOC_to_yolo.py /path/to/input_directory /path/to/output_directory
```

### Running Inference

To perform PPE detection on images:

```
python inference.py /path/to/input_images /path/to/output_images
```

This script will use both the person detection and PPE detection models to process the images.

## Models

The `weights` directory contains two pre-trained models:

1. `person_detection_model.pt`: Used for detecting persons in images
2. `ppe_detection_model.pt`: Used for detecting PPE on the identified persons

These models are used sequentially in the inference process.

## Project Approach

[Approach.pdf](https://github.com/user-attachments/files/16646557/Zyook.1.pdf)  

[Project Explanation Video](https://www.loom.com/share/104b6dec71174eba807c1ed7d1eeaef7?sid=fda79776-f4c9-4666-a265-76d99772bdfc)


