import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import requests
import zipfile
import io

# Create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Download and extract COCO annotations
def download_annotations():
    url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall('data')
    return os.path.join('data', 'annotations', 'instances_train2017.json')

# Download annotations if not already present
annotation_file = os.path.join('data', 'annotations', 'instances_train2017.json')
if not os.path.exists(annotation_file):
    annotation_file = download_annotations()

# Initialize COCO API for instance annotations
coco = COCO(annotation_file)

def download_images(num_images=5):
    image_ids = coco.getImgIds()
    selected_ids = np.random.choice(image_ids, num_images, replace=False)
    image_paths = []

    for img_id in selected_ids:
        img = coco.loadImgs(img_id)[0]
        url = img['coco_url']
        file_name = os.path.join('data', img['file_name'])
        
        if not os.path.exists(file_name):
            response = requests.get(url)
            with open(file_name, 'wb') as file:
                file.write(response.content)
        
        image_paths.append(file_name)
    
    return image_paths

# Load a pretrained YOLOv5 model
model = YOLO('yolov5s.pt')

def detect_objects(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(img)

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            label = f"{model.names[int(cls)]} {conf:.2f}"
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, label, color='white', fontweight='bold', 
                    bbox=dict(facecolor='red', edgecolor='none', alpha=0.7))

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Download images and perform object detection
image_paths = download_images(5)
for path in image_paths:
    print(f"Detecting objects in {path}")
    detect_objects(path)
    print("\n")