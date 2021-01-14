#Importing the required libraries

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random


#Number of workers(devices)
num_of_workers = 100
num_of_classes = 10

#Define a list for each class

lables_list = []
for i in range(num_of_classes):
    lables_list.append([])

#Change the directory to the place of train dataset
os.chdir("D:\Federated_Learning/Applications/Federated_Learning_EigentTrust_100_workers/mnist/datasets/mnist/training/")

#Globbing all the images and filling the classes lists with their images 
path = os.path.join(os.getcwd(), "*.png")
print(path)
filename = ""
for f in glob.glob(path):

    filename = f.split('\\')[-1]
    if "zero" in filename:
        lables_list[0].append(filename)
    elif "one" in filename:
        lables_list[1].append(filename)
    elif "two" in filename: 
        lables_list[2].append(filename)
    elif "three" in filename:
        lables_list[3].append(filename)
    elif "four" in filename:
        lables_list[4].append(filename)
    elif "five" in filename:
        lables_list[5].append(filename)
    elif  "six" in filename:
        lables_list[6].append(filename)
    elif "seven" in filename:
        lables_list[7].append(filename)
    elif "eight" in filename:
        lables_list[8].append(filename)
    elif "nine" in filename:
        lables_list[9].append(filename)
    



for i in range(num_of_classes):
    print(i, ': ')
    print(len(lables_list[i]))

portion_per_class = [] 
for i in range(num_of_classes):
    portion_per_class.append(int(len(lables_list[i])/num_of_workers))
    print(portion_per_class[i])

print(sum(portion_per_class))

#writing images names in text file
print(os.getcwd())
filename = os.path.join(os.getcwd(), "images_names_100.txt")
with open(filename, "w") as fp:
    for i in range(num_of_workers):
        for j in range(num_of_classes):
            fp.writelines("%s\n" % image for image in lables_list[j][i * portion_per_class[j]: (i+1) * portion_per_class[j]])
fp.close()
print("Training distribution done!")








