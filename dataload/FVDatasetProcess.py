import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import sys
import os
from skimage import io,color




# 读取csv文件
class FVDatasetOpen(Dataset):
    def __init__(self, transform = None, set_path=None, csv_path = None):
        pwd = sys.path[0] 
        self.transform = transform
        self.set_path = set_path
        self.test_df = pd.read_csv(csv_path + '/test.csv')

    def __getitem__(self, index):
        img1 = io.imread(self.set_path + self.test_df['img1_path'][index])
        if len(img1.shape) == 2:
            img1 = color.gray2rgb(img1)
        img1_subject = self.test_df['img1_subject'][index]
        img2 = io.imread(self.set_path + self.test_df['img2_path'][index])
        if len(img2.shape) == 2:
            img2 = color.gray2rgb(img2)
        img2_subject = self.test_df['img2_subject'][index]
        label = self.test_df['label'][index]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img1_subject, img2, img2_subject, label
    def __len__(self):
        return self.test_df.shape[0]



