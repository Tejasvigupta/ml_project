#importing the required modules
import numpy as np
import tensorflow as tf
import keras


#define the model
from keras.models import Sequential # to keep stack of convolution layers
from keras.layers import Conv2D     # convolution poeration
from keras.layers import MaxPool2D  # pooling operation for reducing image size
from keras.layers import Dense # classical neural network
from keras.layers import Flatten #to flatten the 2d image

#adding layers to the model
model=Sequential()

input_shape=(64,64,3)

#first convolution
model.add(Conv2D(32,(3,3),input_shape=input_shape,activation='relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

#second convolution
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

#flatten the network
model.add(Flatten())

#Full connection ANN
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

# defining the optimizer
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:\\Users\\intel\\Desktop\\ML-project\\chest_xray\\train',
                                                 target_size = (64, 64),
                                                 batch_size = 16,
                                                 class_mode = 'binary')
validation_set = val_datagen.flow_from_directory('C:\\Users\\intel\\Desktop\\ML-project\\chest_xray\\val',
                                            target_size = (64, 64),
                                            batch_size = 16,
                                            class_mode = 'binary')
model.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 100,
                         validation_data = validation_set,
                         validation_steps = 200)


