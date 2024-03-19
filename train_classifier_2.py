#explanations of the process:
# -- images are present at ./concepts to help better understand the following

#Convolution: (get the set of features for each sub-matrix of pixels)
#in CNN we use a filter over the image and compute the dot product of the image matrix's elements and the filter
# to find the set of feature of the image
# => input image * feature detector = feature map

# Max Pooling:
# if we want to decrease the size of the image, we part the matrix (strange parts) and switch each part with the
# largest number in it and put that in a new decreased image with a smaller size in the corresponding place

#Flattening:
#given a matrix of size nxm we create an array of size 1xmn:
# put all the rows next to each other and then transfer the long row into a long column


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^IMPORTS^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import pickle
import time

#Sequential allows you to create models layer-by-layer for most problems
from keras.models import Sequential

#conv2d helps us create the convolution images
#MaxPooling2D helps us max pool 2d matrixes
#Dense layers are the hidden layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^LOAD DATA^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#in order to read a file that's been saved as a pickle variable we need to use picle.load
#pickle.load( open( filename , 'rb' ) ) - rb is to "read in binary"

IMGS = pickle.load(open('Imgs_Arr.pkl', 'rb'))
CLS = pickle.load(open('Cls_Arr.pkl', 'rb') )
TST = pickle.load(open('Test_Arr.pkl', 'rb') )

#a reminder: IMGS is an array of 3D matrixes, each 3D matrix describes an image,
# each dimension resembles one of the rgb colors that make up that image and each
# matrix is the image's pixels in the corresponding color
# => conclusion : every number in every array is between 0 and 255 because that is the color scale of the pixles

#the lesser the values, the faster the calculations
IMGS = IMGS/255 #convert all numbers from 0 to 255 to numbers between 0 and 1
TST = TST/255
#if we type IMGS.shape we get a tuple containing:
# IMGS.shape = ( number of images, height of images, width of images, number of channels (3) )

# layer by layer model

M = Sequential()

# 1. CONVOLUTION (first time):
#64 convolution layers (64 feature detectors), each feature detector is a 3x3 matrix
#activation function is relu :  in Neural Networks, an Activation Function decides whether a neuron should be activated or not
# the role of rectified linear (ReLU) activation function in CNN:
#ReLU is much faster to compute than sigmoid or tanh, which can make training large neural networks more efficient
#The sigmoid and tanh functions tend to saturate at the extremes of their output range, which can slow down training and prevent the network from learning

M.add( Conv2D( 64, (3,3) , activation= 'relu') )

# 2. MAX POOLING (first time):
# part the convoluted matrix into 2x2 matrixes and choose the max of those parts
M.add( MaxPooling2D( (2,2) ) )

# 3. REDO CONVOLUTION AND MAX POOLING ON THE TINIER MATRIXES FOR ACCURACY:
M.add( Conv2D( 64, (3,3) , activation= 'relu') )
M.add( MaxPooling2D( (2,2) ) )

M.add( Flatten() )

# 4. pass the previous to the neural network and create a dense layer:
#we have 64+64=128 layers (matrixes), input shape = IMGS's shape
#IMGS.shape = (number of images, height, width, channels_num) => input shape = (height, width, channels_num)
M.add( Dense( 128, input_shape= IMGS.shape[1:], activation= 'relu') )

# 5. create the output layer (which is also a dense layer with 2 neurons (cat&dog)):
#M.add( Dense( 2, activation= 'softmax') ) #softmax returns a probability prediction, class a with p1, and class b with 1-p1
M.add( Dense( 1, activation= 'sigmoid') )


# 6. return output
#optimization function:
#Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent
# procedure to update network weights iterative based in training data
#loss function:
#sparse_categorical_crossentropy: Used as a loss function for multi-class classification model where the
# output label is assigned integer value, and your classes are mutually exclusive
# (e.g. when each sample belongs exactly to one class)
# metrics.accuracy:
#Calculates how often predictions equal labels. This metric creates two local variables, total and count
# that are used to compute the frequency with which y_pred matches y_true
#M.compile( optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
M.compile( optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

# 7.Tuning:
#TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow
from keras.callbacks import TensorBoard

# basically: create am object of type (class) TensorBoard
#we want to create a folder in the same directory as this project in which we can create the logs of all the models that
#we are going to train

#in order to do that we should first create a list of names for all the logs
#we want different names so the best way is to use the time as a name
Names = f'cVdPred{int(time.time())}'

tb = TensorBoard(log_dir=f'logs\\{Names}\\')




# 8. run the Model with our data set:
#IMGS = the independent variable
#CLS = the dependent variable
#epoch = the training of the neural network with all the training data for one cycle
#The number of epochs is the number of complete passes through the training dataset.
#validation = a set of examples to test your model (in percent)
#the batch_size is a number of samples processed before the model is updated.
#The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset
M.fit(IMGS, CLS, epochs= 6, validation_split= 0.1, batch_size= 25, callbacks= [tb])


#we get results of type:
# loss: 0.6040 - accuracy: 0.7826 - val_loss: 0.7003 - val_accuracy: 0.5000
#our goal is to get the accuracy and the val_accuracy to be close enough
#after running the code we get a folder called logs which we will now view using tensorborad




# in terminal:
# tensorboard --logdir=logs/
#you'll get a url which you should copy and look up on Google: http://localhost:6006/?smoothing=0.6#timeseries





import numpy as np
import cv2
CATEGORIES = ['dog', 'cat']


def image(path):
    img = cv2.imread(path)
    new_arr = cv2.resize(img, (100, 100))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 100, 100, 3)
    return new_arr

prediction = M.predict([image('test/test1.jpeg')])
print(CATEGORIES[prediction.argmax()])

prediction = M.predict([image('test/test2.jpeg')])
print(CATEGORIES[prediction.argmax()])








