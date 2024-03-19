#after we created a model in ./train_classifier_2.py and we ran the tensorboard command we recieved a graph
# telling us how good our model is
#so we'll create another model with an extra convolution layer and an extra dense layer
# to see if the new model is any different/better



import pickle
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMGS = pickle.load(open('Imgs_Arr.pkl', 'rb'))
CLS = pickle.load(open('Cls_Arr.pkl', 'rb') )

IMGS = IMGS/255 #convert all numbers from 0 to 255 to numbers between 0 and 1

M = Sequential()

# 1. CONVOLUTION (first time):

M.add( Conv2D( 64, (3,3) , activation= 'relu') )

# 2. MAX POOLING (first time):
M.add( MaxPooling2D( (2,2) ) )

# 3. REDO CONVOLUTION AND MAX POOLING ON THE TINIER MATRIXES FOR ACCURACY:
M.add( Conv2D( 64, (3,3) , activation= 'relu') )
M.add( MaxPooling2D( (2,2) ) )

#*******************************************EXTRA*******************************************************
M.add( Conv2D( 64, (3,3) , activation= 'relu') )
M.add( MaxPooling2D( (2,2) ) )
#*******************************************************************************************************

M.add( Flatten() )

# 4. pass the previous to the neural network and create a dense layer:
M.add( Dense( 128, input_shape= IMGS.shape[1:], activation= 'relu') )

#*******************************************EXTRA*******************************************************
M.add( Dense( 128, input_shape= IMGS.shape[1:], activation= 'relu') )
#*******************************************************************************************************

# 5. create the output layer (which is also a dense layer with 2 neurons (cat&dog)):
M.add( Dense( 2, activation= 'softmax') ) #softmax returns a probability prediction, class a with p1, and class b with 1-p1

# 6. return output
M.compile( optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])


# 7.Tuning:
from keras.callbacks import TensorBoard

Names = f'cVdPred{int(time.time())}'

tb = TensorBoard(log_dir=f'logs\\{Names}\\')




# 8. run the Model with our data set:
M.fit(IMGS, CLS, epochs= 6, validation_split= 0.1, batch_size= 5, callbacks= [tb])


#we get results of type:
# loss: 0.6572 - accuracy: 0.5870 - val_loss: 0.7334 - val_accuracy: 0.5000
#our goal is to get the accuracy and the val_accuracy to be close enough
#after running the code we get a folder called logs which we will now view using tensorborad


# in terminal:
# tensorboard --logdir=logs/
#you'll get a url which you should copy and look up on Google: http://localhost:6006/#timeseries

#from the graph we recieved we got that the first model was better because it inhances accuracy while decreasing loss

#i accedentally messed up the first model so now this one is better haha





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

