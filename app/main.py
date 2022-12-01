from fastapi import FastAPI, File
import torch
import cv2
import numpy as np

# custom file
from app.get_boundingbox import inference, image_preprocessing 
from app.get_text import ocr_api


def run(image):
    # print(os.)

    # load YOLOv5 with custom weight
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='Model_API/YOLOv5 Weight/custom_221117.pt')

    s,img = inference(model, image) # 유통기한 부분만 크롭
    
    if s: # bounding box 찾은 경우
        pre_img = image_preprocessing(img) # 글자 잘 인식하도록 전처리
        text = ocr_api(pre_img) # 이미지에서 텍스트 추출
        return text, img

    else: # bounding box 찾지 못한 경우
        print("인식하지 못했습니다")
        return None, None


app = FastAPI()

@app.post("/exp_date")
def predict_ExpDate(file: bytes = File()):

    decoded = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR) # BGR

    result, cropped_image = run(decoded)
    
    #이미지 확인 코드 (이미지 창을 닫아야 그 다음이 진행된다.)
    if result != None:
        cv2.imshow('image', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(result)

    return {'exp_date': result}


@app.post("/test")
def test(file: bytes = File()):
    decoded = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(f"test.jpg", decoded)