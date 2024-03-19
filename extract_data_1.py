#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^IMPORTS^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#numpy is for array operations
import numpy as np

#opencv is a library of Python bindings designed to solve computer vision problems (aka images)
#cv2 is for converting images into arrays
import cv2

#os is to specify the location of images
import os

import random

#matplotlib.pyplot is a collection of functions that make matplotlib work like MATLAB
import matplotlib.pyplot as plt

#pickle is to save data
import pickle

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^DIRECTORY^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# the r before a string tells python to treat it as a raw string so that the backslashes are not escape characters
Directory = r'C:\Users\asus\Desktop\project\dataset'

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Classifiers^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#the classifiers are Cat/0, Dog/1
CLASSIFIERS = ['dog', 'cat']

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^EXAMPLE SET^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#extract the examples from the directory and turn them into arrays

#we need all the images to be of a unified size
img_size = 100 #less pixels mean a more blurred image and more pixels mean a longer computation time

data_set = []

#for each classifier (cat/dog)
for c in CLASSIFIERS:
    cur_classifier_directory = os.path.join(Directory, c)
    classifier = CLASSIFIERS.index(c)
    #for each example image for that specific classifier:
    for img in os.listdir(cur_classifier_directory):
        #os.listdir(path) goes through all directories in path, aka all the images in a folder
        img_directory = os.path.join(cur_classifier_directory, img)
        img_arr1 = cv2.imread(img_directory) #convert image to array
        #resize the image array:
        img_arr = cv2.resize(img_arr1, (img_size, img_size))
        data_set.append([img_arr, classifier])


#we need to shuffle the data because now we have all the cats first and dogs second
#the probability of finding an image's match to be a cat is higher than it is a dog's
#so shuffeling the data gives a more unified probability

random.shuffle(data_set)
random.shuffle(data_set)
random.shuffle(data_set)


#put all the image arrays together and all their classifiers togrther (in order of the images)
imgs = []
cls = [] #short for classifers

for m, c in data_set:
    imgs.append(m)
    cls.append(c)

#convert lists into arrays:
imgs_arr = np.array(imgs)
cls_arr = np.array(cls)

#save the data above to the computer using pickle
#pickle.dump(obj, file) - to save data as a pickle variable
#open(filename, wb) means open a file with the name filename and wb means write in binary

pickle.dump(imgs_arr, open('Imgs_Arr.pkl', 'wb'))
pickle.dump(cls_arr, open('Cls_Arr.pkl', 'wb'))




#******************************************************************************************
# the following is unnecessary but im scared to delete it
#do the same for the test folder:
TestDirectory = r'C:\Users\asus\Desktop\project\test'
test_imgs = []

for img in os.listdir(TestDirectory):
    # os.listdir(path) goes through all directories in path, aka all the images in a folder
    img_directory = os.path.join(TestDirectory, img)
    img_arr1 = cv2.imread(img_directory)  # convert image to array
    # resize the image array:
    img_arr = cv2.resize(img_arr1, (img_size, img_size))
    test_imgs.append(img_arr)

test_arr = np.array(test_imgs)
pickle.dump(imgs_arr, open('Test_Arr.pkl', 'wb'))
