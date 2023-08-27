import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

argA = 'pidnet-s'
argC = True
argP = './pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt'
argR = './samples/'
argT = '.png'


model = models.pidnet.get_pred_model(argA, 19 if argC else 11)
model = load_pretrained(model, argP).cuda()
model.eval()

PERSON_CLASS_INDEX = 11
THRESHOLD = 0.5

def detect_seg(imgdata, debug=False):
    with torch.no_grad():
        # PERSON_CLASS_INDEX = 11

        if debug:
            cv2.imwrite(f'webcam before seg.jpg', imgdata)

        # 이미지 전처리
        img = input_transform(imgdata)
        img = img.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        # AI 모델 추론
        logits = model(img)
        logits = F.interpolate(logits, size=img.size()[-2:], mode='bilinear', align_corners=True)
        probs = F.softmax(logits, dim=1)  # 클래스별 확률 계산
        max_probs, preds = torch.max(probs, dim=1)  # 최대 확률과 해당 클래스 인덱스

        preds_np = preds.cpu().numpy().squeeze()

        # 확률 임계값을 사용하여 마스크 생성
        mask = (max_probs > THRESHOLD).cpu().numpy().squeeze()

        # 특정 클래스 (예: 사람)와 확률 임계값을 만족하는 픽셀만 선택
        person_mask = (preds_np == PERSON_CLASS_INDEX) & mask
        output_img = np.zeros_like(imgdata)
        output_img[person_mask] = imgdata[person_mask]

        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        if debug:
            cv2.imwrite(f'webcam after seg.jpg', output_img)

        return output_img
