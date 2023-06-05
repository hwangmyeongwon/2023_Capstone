import cv2
import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import time
import torch.optim as optim

from tqdm import tqdm



file_path = 'C:/capstone/face_detect/images/faces/train/*/*.jpg'
file_list = glob(file_path)
print(len(file_list))#총 데이터의 갯수

data_dict = {'image_name':[],'class':[],'target':[], 'file_path':[]}
target_dict = {'positive':0,'negative':1,'neutral':2}

for path in file_list:
    data_dict['file_path'].append(path)  # file_path 항목에 파일 경로 저장
    path_list = path.split(os.path.sep)  # os별 파일 경로 구분 문자로 split
    print(path_list)
    data_dict['image_name'].append(path_list[-1]) #
    data_dict['class'].append(path_list[-2]) #
    data_dict['target'].append(target_dict[path_list[-2]]) #

train_df = pd.DataFrame(data_dict)
print('\n<data frame>\n', train_df)
print(train_df.to_csv("./emotion.csv", mode='w'))

#여기부터 모델 학습

from sklearn.model_selection import train_test_split
def get_df():
    df = pd.read_csv('./emotion.csv')
    df_train, df_test = train_test_split(df,test_size=0.2,random_state=1000)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=1000)

    return df_train, df_val,df_test

df_train,df_val,df_test = get_df()
print(len(df_train), len(df_val),len(df_test))

import torch
from torch.utils.data import Dataset


class Classification_Dataset(Dataset):
    def __init__(self,csv,mode,transform=None):
        self.csv = csv.reset_index(drop=None)#인덱스를 reset 시키고 다시 부여
        self.transform = transform#transform시킴 - 데이터 개수 늘리기
    def __len__(self):
        return self.csv.shape[0]#데이터 개수

    def __getitem__(self,index):
        row = self.csv.iloc[index]
        image = Image.open(row.file_path).convert('RGB')#파일을 열어 이미지 읽고 rgb 변환
        target = torch.tensor(self.csv.iloc[index].target).long()#
        #torch.tensor는 data를 tensor로 copy해주는 것
        #iloc가 전체 데이터 프레임에서 index번째 행에 있는 값만 추출해라
        if self.transform:
            image = self.transform(image)#transform 적용

        return image,target

dataset_train = Classification_Dataset(df_train, 'train', transform=transforms.ToTensor())

# 데이터(shape:torch.Size([3, 381, 343])) rgb에 대한 mean, std 구하기 #batch normalization --반드시 해야하는 것 함
rgb_mean = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset_train]
rgb_std = [np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset_train]

# 각 데이터 채널별로 mean, std 나타내기
c_mean = []
c_std = []

for i in range(3):
    c_mean.append(np.mean([m[i] for m in rgb_mean]))
    c_std.append(np.std([s[i] for s in rgb_std]))

print(f'rgb mean: {c_mean}\nrgb std: {c_std}')



def get_transforms(image_size):
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.4222492, 0.36672342, 0.3263919],
                             [0.043753106, 0.047309674, 0.05365861])])

    transforms_val = transforms.Compose([
                                         transforms.Resize(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.4222492, 0.36672342, 0.3263919],
                                                              [0.043753106, 0.047309674, 0.05365861])])

    return transforms_train, transforms_val

transforms_train, transforms_val = get_transforms(224)

dataset_train = Classification_Dataset(df_train, 'train', transform=transforms_train)
dataset_val = Classification_Dataset(df_val, 'valid', transform=transforms_val)

from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, sampler=RandomSampler(dataset_train), num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=32, num_workers=0)

from torchvision import models
from collections import OrderedDict
import torch.nn as nn

device = 'cpu' #모델을 cuda로 바꾸는 코드 중요 중요 중요
if torch.cuda.is_available():
    device = 'cuda'

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=3, in_channels=3)
# model = models(pretrained=True)
# import timm
# model = timm.create_model('efficientnet_b0', pretrained=True)
# print(model)

model = models.vgg16(pretrained=True)
# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b3')
print(model)

for param in model.parameters():
    param.requires_grad = False

# 마지막 layer를 과제에 맞게 수정하기
classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 256)),
            ('relu', nn.ReLU()),
            ('drop',nn.Dropout(0.2)),
            ('fc2', nn.Linear(256, 3))
            ]))

model.classifier = classifier

def train_epoch(model, loader, device, criterion, optimizer):
    model.train()  # 모델 train 모드로 바꾸기
    train_loss = []
    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):
        optimizer.zero_grad()  # 최적화된 모든 변수 초기화

        data, target = data.to(device), target.to(device)  # 지정한 device로 데이터 옮기기
        logits = model(data)  # 1. forward pass

        loss = criterion(logits, target)  # 2. loss 계산
        loss.backward()  # 3. backward pass

        optimizer.step()  # 4. gradient descent(파라미터 업데이트)

        loss_np = loss.detach().cpu().numpy()  # loss값 가져오기 위해 gpu에 있던 데이터 모두 cpu로 옮기기
        train_loss.append(loss_np)
        bar.set_description('loss: %.5f' % (loss_np))

    train_loss = np.mean(train_loss)  # 한 epoch당 train loss의 평균 구하기
    return train_loss


def val_epoch(model, loader, device, criterion):
    model.eval()  # 모델 evaluate 모드로 바꾸기
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)  # 지정한 device로 데이터 옮기기
            logits = model(data)  # 1. forward pass
            probs = logits.softmax(1)  # 다중분류 -> 각 클래스일 확률을 전체 1로 두고 계산하기

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)  # 2. loss 계산
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    # accuracy : 정확도
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

    return val_loss, acc

def run(model, init_lr, n_epochs):
    # gpu 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model을 지정한 장치로 옮기기
    model = model.to(device)

    # loss function 지정
    criterion = nn.CrossEntropyLoss()

    # optimizer로 adam 사용
    optimizer = optim.Adam(model.parameters(), lr=init_lr,weight_decay=0.4)#regularization적용

    for epoch in range(1, n_epochs + 1):
        print(time.ctime(), f'Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, device, criterion, optimizer)  # train
        val_loss, acc = val_epoch(model, valid_loader, device, criterion)  # validation

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, Acc: {(acc):.4f}.'
        print(content)

    torch.save(model, 'best_model_5.pth')
    torch.save(model.state_dict(), 'best_model_state_dict5.pth')


run(model, init_lr=4e-6, n_epochs=30)
# import torch
# device = 'cpu' #모델을 cuda로 바꾸는 코드 중요 중요 중요
# if torch.cuda.is_available():
#     device = 'cuda'
# print(device)
