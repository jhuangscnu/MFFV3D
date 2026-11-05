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



class GetTrainImage(Dataset):
    def __init__(self, path_db=None, path_csv=None, transform=None, csv_name=None, postfix=None):
        self.path_db = path_db
        self.transform = transform
        self.csv_name = csv_name
        self.postfix = postfix
        self.path_csv = os.path.join(path_csv, self.csv_name)
        self.csv = pd.read_csv(self.path_csv)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        ref_id = self.csv['name'][index]
        id = int(ref_id[:4])
        if id >2000:
            id = id - 1374
        path_ref_id_img = self.path_db + '/' + ref_id + self.postfix
        img_ref_id = Image.open(path_ref_id_img).convert('RGB')
        img_ref_id = np.array(img_ref_id)
        if self.transform:
            img_ref_id = self.transform(img_ref_id)
        return img_ref_id, id
        

class GetAuthenticationPair(Dataset):
    def __init__(self, path_db=None, path_csv = None, transform = None, csv_name = None, postfix = '.bmp'):
        self.path_db = path_db
        self.transform = transform
        self.csv_name = csv_name
        self.postfix = postfix
        self.path_csv = os.path.join(sys.path[0], path_csv, self.csv_name)
        self.csv = pd.read_csv(self.path_csv)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        bio_ref_subject_id = self.csv['name1'][index]
        id1 = int(bio_ref_subject_id[:4])
        if id1 >2000:
            id1 = id1 - 1374

        probe_subject_id = self.csv['name2'][index]
        id2 = int(probe_subject_id[:4])
        if id2 >2000:
            id2 = id2 - 1374

        label = self.csv['label'][index]

        path_ref_id_img1 = self.path_db + '/' + bio_ref_subject_id + self.postfix
        img1_ref_id = Image.open(path_ref_id_img1).convert('RGB')
        img1_ref_id = np.array(img1_ref_id)
        if self.transform:
            img1_ref_id = self.transform(img1_ref_id)
        
        path_ref_id_img2 = self.path_db + '/' + probe_subject_id + self.postfix
        img2_ref_id = Image.open(path_ref_id_img2).convert('RGB')
        img2_ref_id = np.array(img2_ref_id)
        if self.transform:
            img2_ref_id = self.transform(img2_ref_id)

        return img1_ref_id, img2_ref_id, id1, id2, label

