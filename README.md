<img src="frontend/AInterview.gif" width="456" height="132">

# AInterview
GAN(AI)으로 생성한 가상 면접관과 면접을 보고 AI 분석을 통해 피드백을 받아보는 면접 준비 웹사이트

<br>

# 프로젝트 소개
취업 시 면접은 필수적인 요소로 작용됩니다. 그래서 집에서도 면접연습을 할 수 있도록 면접관을 StyleGAN 과 Faceswap 으로 생성하고,대화는 openai API를 이용해 챗봇으로 구현하여 실제 면접을 진행하는 느낌을 주어서 현실감이 있는 면접연습을 할 수 있도록 프로젝트르 진행하였습니다.
<br><br>
면접이 다 끝나면 시선처리,머리움직임,목소리 크기,표정변화 에 대해 AI를 이용하여 피드백을 제공해줍니다.
<br><br>
집에서도 누구나 이용할 수 있도록 노트북의 웹캠을 이용하여 사용자를 모습을 담을려는 취지로 웹사이트를 제작하였고,AI와 웹사이트를 연결시키기 위해 Flask를 이용하였습니다.

<br>

# architecture
<img  alt="image" src="/AInterview_architecture/AInterview_architecture.png">

<br>

## 팀원 구성

|이름|전공|담당분야|
|---|---|------|
|이승연|빅데이터전공|백엔드|
|황명원|콘텐츠IT전공|프론트엔드|
|이다해|빅데이터전공|AI|
|임수빈|스마트IoT전공|AI|

<br>

## 개발에 쓰인 기술


### AI
- PyTorch
- MediaPipe
- matplotlib



### BackEnd
- Flask
- MySQL



### FrontEnd
- html
- css
- javascript
- bootstrap


<br>

## 팀 회의 일지
https://www.notion.so/AInterview-6ec93ec6d9744fab83edd0b8ab9a9714?pvs=4


<br>

## 참고 자료 및 개발 과정
* [프론트엔드 참고자료](https://www.notion.so/366079d1184c42b3aa9f0797c642b75e?pvs=4)
* [백엔드 참고자료](https://www.notion.so/4115812c39134336aba01132b20aa8c9?pvs=4)
* [챗봇 개발과정](https://www.notion.so/ChatGPT-990ab9ccb89243a5be1188a0d177b592?pvs=4)
* [가상 면접관 개발과정](https://www.notion.so/c0c2564d54cd4ccb97b61223865da39b?pvs=4)
* 피드백 부분(AI)
  1. [머리 움직임](https://www.notion.so/963dd322c4cf4097a35d4422ac592984?pvs=4)
  2. [표정 감지](https://www.notion.so/58c42d4cf0554e14b035e3d5659b5a7d?pvs=4)
  3. [시선 처리](https://www.notion.so/7ecf2c7bc18c4042967ba02e8821aec1?pvs=4)
  4. [목소리 크기](https://www.notion.so/f7ddd7d53ab942338ca50568d529a559?pvs=4)
