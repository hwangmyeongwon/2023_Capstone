import cv2
import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from mtcnn import MTCNN
from facenet_pytorch import MTCNN
import time

from PIL import Image
from torchvision.utils import save_image
import torch
import torchvision.models as models
import torchvision
model = models.vgg16()
emotion_model = torch.load('./best_model_1.pth', map_location=torch.device('cpu')) #학습이 완료된 모델과 가중치
emotion_model.eval()
# sex_model = torch.load('./sex_model_2.pth', map_location=torch.device('cpu')) #학습이 완료된 모델과 가중치
# sex_model.eval()


target_dict = {'positive':0,'neutral':1,'negative':2}
target1 = ['positive','neutral','negative']

# target_dict = {'male':0,'female':1}
# target = ['male','female']


capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector = MTCNN()

while True:
    # time.sleep(0.5)
    ret, frame = capture.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_res = detector.detect(img)
    print(det_res[1])

    try:
        len(det_res[0])
    except:
        continue

    if det_res :
        for face in det_res[0]: #좌표즐만 가지고 옴
            x1, x2, y1, y2 = list(map(int,face))#좌표를 각각 float에서 int로
            print(x1,x2,y1,y2)
            center1 =int(x1 + (y1 - x1)/2)
            center2 = int(x2 + (y2 - x2)/2)

            a = int(max(center1 - x1,center2 - x2))

            cv2.rectangle(frame, (center1-a, center2-a), (center1+a, center2+a), (0, 0, 255), 2)  # 사각형
            #else:
                #cv2.rectangle(frame, (x1, x2),(y1+a, y2+a),(0, 0, 255),2)
            cv2.imwrite("asd.png",frame)#
            frame1 = frame[center2-a:center2+a, center1-a:center1+a]#경계 안으로 자르기
            try:
                cv2.imwrite("asd1.png", frame1)
                frame1 = cv2.resize(frame1,(48, 48))  # 사이즈 조정
                cv2.imwrite("asd2.png", frame1)
            except:
                cv2.imshow('Face', frame)
                cv2.waitKey(1)
                continue

            cv2.imshow('Face',frame)
            cv2.waitKey(1)

            with torch.no_grad(): # (1,3,48,48)
                input_Data = np.reshape(frame1, ((1,) + frame1.shape)) # (48,48,3) -> (1, 48, 48, 3)
                input_Data = torch.Tensor(input_Data).permute(0, 3, 1, 2) # (1, 48, 48, 3) -> (1, 3, 48, 48))
                # print(input_Data.shape)

                outputs = emotion_model(input_Data)
                e_p = torch.argmax(outputs)
                print(target1[e_p.data])
                #
                # outputs = sex_model(input_Data)
                # e_p = torch.argmax(outputs)
                # print(target[e_p.data])


    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()

