# 추론 결과로 나온 박스 모두 확인

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import torch
from app.get_boundingbox import inference

model = torch.hub.load('ultralytics/yolov5', 'custom', path='YOLOv5 Weight/custom_221117.pt')

image = cv2.imread('for-retraining/garbages/IMG_4510.JPG')
nob, boxes = inference(model, image)
box_images = [x[1] for x in boxes]

for img in box_images:
    cv2.imshow('inference result', img)
    cv2.waitKey()
    cv2.destroyAllWindows()