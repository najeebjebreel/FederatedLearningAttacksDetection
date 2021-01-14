import os
import torch
import numpy as np
from skimage.color import rgb2gray
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms

import PIL
from PIL import Image

class Dataset(data.Dataset):
    
    def __init__(self, data_dir, transform, start_idex=0, end_index=50000, image_size=28,  split_dataset = False):
        self.data_dir = data_dir
        self.transform = transform
        # if "training" in data_dir:
        #     print("=======> loading training dataset......\n")
        # else:
        #     print("=======> loading testing dataset......\n")
        
        self.image_size = image_size
        self.start_index = start_idex
        self.end_endex = end_index
        self.current_worker_dataset_portion = []
        self.lables = {"zero":0, 
        "one":1,
        "two":2,
        "three":3,
        "four":4,
        "five":5,
        "six":6,
        "seven":7,
        "eight":8,
        "nine":9}
      
        with open(os.path.join(data_dir, "images_names_100.txt")) as fp:
            self.images = [line.strip() for line in fp]
        if(split_dataset==True):
            self.current_worker_dataset_portion = self.images[self.start_index:self.end_endex]
        else:
            self.current_worker_dataset_portion = self.images
        
        
        self.images = self.current_worker_dataset_portion
        
    def __getitem__(self, index):
        
        img_name = self.images[index]
        
        #print(os.path.join(self.data_dir, img_name))
        img = Image.open(os.path.join(self.data_dir, img_name))
        img = img.resize((self.image_size,self.image_size))

        img_label = img_name.split("(")[0].strip()
        label = self.lables[img_label]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label #, img_name
    
    def __len__(self):
        return len(self.images)
      