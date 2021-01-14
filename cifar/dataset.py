import os
import torch
import numpy as np
from skimage.color import rgb2gray
import numpy as np
#import cv2
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms

import PIL
from PIL import Image

class Dataset(data.Dataset):
    
    def __init__(self, data_dir, transform, start_index, end_index, image_size,  split_dataset):
        self.data_dir = data_dir
        self.transform = transform
        if "training" in data_dir and split_dataset == False:
            print("=======> loading training dataset......\n")
            
        if 'testing' in data_dir:
            print("=======> loading testing dataset......\n")
        
        self.image_size = image_size
        self.start_index = start_index
        self.end_index = end_index
        self.current_worker_dataset_portion = []
        self.lables = {"airplane":0, 
        "automobile":1,
        "bird":2,
        "cat":3,
        "deer":4,
        "dog":5,
        "frog":6,
        "horse":7,
        "ship":8,
        "truck":9}

      
        with open(os.path.join(data_dir, "images_names.txt")) as fp:
            self.images = [line.strip() for line in fp]
        if(split_dataset==True):
            self.current_worker_dataset_portion = self.images[self.start_index:self.end_index]
        else:
            self.current_worker_dataset_portion = self.images
        
        self.images = self.current_worker_dataset_portion
        
    def __getitem__(self, index):
        
        img_name = self.images[index]
        
        #print(os.path.join(self.data_dir, img_name))
        img = Image.open(os.path.join(self.data_dir, img_name))
        img = img.resize((self.image_size,self.image_size))

        img_label = img_name.split("_")[-1].split(".")[0]
       
        label = self.lables[img_label]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label #, img_name
    
    def __len__(self):
        return len(self.images)
      