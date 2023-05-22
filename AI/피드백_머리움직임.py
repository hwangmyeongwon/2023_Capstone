#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import time
# 웹캠 연결
cap = cv2.VideoCapture(0)
a = []
g = []
# 머리 움직임 판단 함수
def check_head_movement(prev_frame, cur_frame):
    # 이전 프레임과 현재 프레임을 그레이 스케일ㅂ로 변환
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    # 이전 프레임과 현재 프레임의 차이 계산
    diff = cv2.absdiff(prev_gray, cur_gray)
    a.append(np.mean(diff))
    return False

# 이전 프레임 초기화
_, prev_frame = cap.read()

# 루프 시작
while True:
    # 현재 프레임 읽기
    ret, cur_frame = cap.read()
    time.sleep(1)
    if not ret:
        break

#     머리 움직임 판단
    if check_head_movement(prev_frame, cur_frame):
        print("감지")

    # 이전 프레임 업데이트
    prev_frame = cur_frame #현재가 이전이 되므로 

    # 화면 출력
    cv2.imshow('frame', cur_frame)

    # 종료 키
    if cv2.waitKey(1) == ord('q'):
        break
        
import matplotlib.pyplot as plt
sum = 0
count = 0
for i in a:
    sum = sum + i
    count = count + 1
    if count == 5:
        re = sum / 10
        g.append(re)
        sum = 0
        count  = 0
        
plt.xlabel('Time(s)')
plt.ylabel('head movement')
plt.grid()
plt.plot(g)    
plt.savefig('head_movement.jpg', facecolor='white')
# 리소스 해제
cap.release()
cv2.destroyAllWindows()


# In[ ]:




