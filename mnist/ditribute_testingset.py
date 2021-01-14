#Importing the required libraries

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random


#Number of workers(devices)
num_of_workers = 10

#Define a list for each class

test_zero_list = []
test_one_list = [] 
test_two_list = []
test_three_list = [] 
test_four_list = [] 
test_five_list = [] 
test_six_list = [] 
test_seven_list = [] 
test_eight_list = [] 
test_nine_list = [] 







#Change the directory to the place of test dataset
os.chdir("d:/Federated_Learning/Applications/mnist/datasets/testing/")



#Globbing all the images and filling the classes lists with their images 
path = os.path.join(os.getcwd(), "*.png")
print(path)
filename = ""
for f in glob.glob(path):

    filename = f.split('\\')[-1]

    if "zero" in filename:
        test_zero_list.append(filename)
    elif "one" in filename:
        test_one_list.append(filename)
    elif "two" in filename: 
        test_two_list.append(filename)
    elif "three" in filename:
        test_three_list.append(filename)
    elif "four" in filename:
        test_four_list.append(filename)
    elif "five" in filename:
        test_five_list.append(filename)
    elif  "six" in filename:
        test_six_list.append(filename)
    elif "seven" in filename:
        test_seven_list.append(filename)
    elif "eight" in filename:
        test_eight_list.append(filename)
    elif "nine" in filename:
        test_nine_list.append(filename)
    

print(len(test_zero_list))
print(len(test_one_list))
print(len(test_two_list))
print(len(test_three_list))
print(len(test_four_list))
print(len(test_five_list))
print(len(test_six_list))
print(len(test_seven_list))
print(len(test_eight_list))
print(len(test_nine_list))



portion_per_class = 98
#writing images names in text file
print(os.getcwd())
filename = os.path.join(os.getcwd(), "images_names.txt")
with open(filename, "w") as fp:
    for i in range(num_of_workers-1):
        fp.writelines("%s\n" % image for image in test_zero_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in test_one_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in test_two_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in test_three_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in test_four_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in test_five_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in test_six_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in test_seven_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in test_eight_list[i * portion_per_class:i * portion_per_class + portion_per_class])
        fp.writelines("%s\n" % image for image in test_nine_list[i * portion_per_class:i * portion_per_class + portion_per_class])
    
#the rest of images for the last worker
    i = 9
    fp.writelines("%s\n" % image for image in test_zero_list[i * portion_per_class:])
    fp.writelines("%s\n" % image for image in test_one_list[i * portion_per_class:])
    fp.writelines("%s\n" % image for image in test_two_list[i * portion_per_class:])
    fp.writelines("%s\n" % image for image in test_three_list[i * portion_per_class:])
    fp.writelines("%s\n" % image for image in test_four_list[i * portion_per_class:])
    fp.writelines("%s\n" % image for image in test_five_list[i * portion_per_class:])
    fp.writelines("%s\n" % image for image in test_six_list[i * portion_per_class:])
    fp.writelines("%s\n" % image for image in test_seven_list[i * portion_per_class:])
    fp.writelines("%s\n" % image for image in test_eight_list[i * portion_per_class:])
    fp.writelines("%s\n" % image for image in test_nine_list[i * portion_per_class:])
fp.close()
print("Testing distribution done!")








