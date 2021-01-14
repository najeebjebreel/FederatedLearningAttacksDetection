#Importing the required libraries

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random


#Number of workers(devices)
num_of_workers = 100

#Define a list for each class

trainairplane_list = []
trainautomobile_list = [] 
trainbird_list = [] 
traincat_list = [] 
traindeer_list = [] 
traindog_list = [] 
trainfrog_list = [] 
trainhorse_list = [] 
trainship_list = [] 
traintruck_list = []


#Change the directory to the place of train dataset
os.chdir("D:\Federated_Learning\Applications\cifar_classification_model\dataset\\train/")

#Globbing all the images and filling the classes lists with their images 
path = os.path.join(os.getcwd(), "*.png")
filename = ""
for f in glob.glob(path): 
    filename = f.split()[-1].split("\\")[-1]
    if "airplane" in filename:
        trainairplane_list.append(filename)
    elif "automobile" in filename:
        trainautomobile_list.append(filename)
    elif "bird" in filename: 
        trainbird_list.append(filename)
    elif "cat" in filename:
        traincat_list.append(filename)
    elif "deer" in filename:
        traindeer_list.append(filename)
    elif  "dog" in filename:
        traindog_list.append(filename)
    elif "frog" in filename:
        trainfrog_list.append(filename)
    elif "horse" in filename:
        trainhorse_list.append(filename)
    elif "ship" in filename:
        trainship_list.append(filename)
    elif "truck" in filename:
        traintruck_list.append(filename)

    

#shuffeling the images of each class

random.shuffle(trainairplane_list)
random.shuffle(trainautomobile_list) 
random.shuffle(trainbird_list)
random.shuffle(traincat_list)
random.shuffle(traindeer_list)
random.shuffle(traindog_list)
random.shuffle(trainfrog_list)
random.shuffle(trainhorse_list) 
random.shuffle(trainship_list) 
random.shuffle(traintruck_list)

print(len(trainairplane_list))
print(len(trainautomobile_list))
print(len(trainbird_list))
print(len(traincat_list))
print(len(traindeer_list))
print(len(traindog_list))
print(len(trainfrog_list))
print(len(trainhorse_list))
print(len(trainship_list))
print(len(traintruck_list))


portion_per_class = int(len(trainairplane_list)/num_of_workers)
#writing images names in text file
filename = os.path.join(os.getcwd(), "images_names.txt")
with open(filename, "w") as fp:
    for i in range(num_of_workers):
        fp.writelines("%s\n" % image for image in trainairplane_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in trainautomobile_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in trainbird_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in traincat_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in traindeer_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in traindog_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in trainfrog_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in trainhorse_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in trainship_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in traintruck_list[i * portion_per_class:i * portion_per_class + portion_per_class])

fp.close()
print("Done!")




