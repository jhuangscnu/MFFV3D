import os
import sys
from PIL import Image
from PIL import ImageDraw
import cv2 as cv
import numpy as np
import math
import shutil
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import random


# base on the results of FVIA, FinSeg, runrunduan, and RecDec
class Get3DUnfoldTrainImageDirectly(Dataset):
    def __init__(self, path_db=None, path_csv=None, transform=None, csv_name=None, postfix=None, angle=1):
        self.path_db = path_db
        self.transform = transform
        self.csv_name = csv_name
        self.postfix = postfix
        self.path_csv = os.path.join(path_csv, self.csv_name)
        self.csv = pd.read_csv(self.path_csv)
        self.angle = angle

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        id = self.csv['finger'][index]
        img = str(self.csv['finger'][index]) + '_' + str(self.angle) + '_' + str(self.csv['sample'][index])
        if img == '84_1_8':
            img = str(self.csv['finger'][index]) + '_' + str(self.angle) + '_' + random.choice(['1', '2', '3', '4', '5', '6', '7', '9', '10'])
        path_img = self.path_db + '/' + img + self.postfix
        img = Image.open(path_img)#.convert('L')
        img = np.array(img)
        if self.transform:
            img = self.transform(img)

        return img, id




# base on the results of FVIA, FinSeg, runrunduan, and RecDec
class Get3DUnfoldAuthenticationPairDirectly(Dataset):
    def __init__(self, path_db=None, path_csv = None, transform = None, csv_name = None, postfix = '.bmp', angle=1):
        self.path_db = path_db
        self.transform = transform
        self.csv_name = csv_name
        self.postfix = postfix
        self.path_csv = os.path.join(sys.path[0], path_csv, self.csv_name)
        self.csv = pd.read_csv(self.path_csv)
        self.angle = angle

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        bio_ref_subject_id = self.csv['finger_enrolled'][index]
        probe_subject_id = self.csv['finger_probe'][index]
        label = self.csv['label'][index]
        bio_ref_reference_id = str(self.csv['finger_enrolled'][index]) + '_' + str(self.angle) + '_' + str(self.csv['sample_enrolled'][index])
        probe_reference_id = str(self.csv['finger_probe'][index]) + '_' + str(self.angle) + '_'  + str(self.csv['sample_probe'][index])
        path_img1 = self.path_db + '/' + bio_ref_reference_id + self.postfix
        path_img2 = self.path_db + '/' + probe_reference_id + self.postfix
        img1 = Image.open(path_img1)# .convert('RGB')
        img1 = np.array(img1)
        img2 = Image.open(path_img2)# .convert('RGB')
        img2 = np.array(img2)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
    
        return img1, img2, bio_ref_subject_id, probe_subject_id, label