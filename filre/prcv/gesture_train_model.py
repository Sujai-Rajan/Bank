# Sujai Rajan

# Final Project: Gesture Recognition System

# CS5330 - Pattern Recognition and Computer Vision


# import statements
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array , array_to_img , load_img
import keras
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Activation,Dropout
# from tensorflow.keras.layers.advanced_activations import LeakyReLU 
 
from keras.layers.activation import LeakyReLU


train_path = './image_dataset/train'
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range = 4,
                                   width_shift_range=0.15,
                                   height_shift_range = 0.2,
                                   shear_range=0.3,
                                   fill_mode='nearest',
                                   validation_split=0.25)
train_set = train_datagen.flow_from_directory(directory=train_path, class_mode='categorical',
                                                    color_mode = 'grayscale',
                                                    target_size=(64,64), batch_size=128, shuffle=True,
                                                    subset ="training")
validation_set = train_datagen.flow_from_directory(directory=train_path,
                                                 target_size = (64,64),
                                                 batch_size = 64,
                                                 class_mode = 'categorical',
                                                 color_mode = 'grayscale',subset="validation",shuffle=True)


def make_model():
    model = Sequential()
    model.add(Conv2D(32,input_shape=(64,64,1),kernel_size=(3,3),strides=(1,1),activation='relu'))
    padding="same"
    model.add(Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='relu'))
    padding="same"
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.15))
    

    model.add(Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='relu'))
    padding="same"
    model.add(Conv2D(128,kernel_size=(3,3),strides=(1,1),activation='relu'))
    padding="same"
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(18,activation='softmax'))
 
    
    model.compile(
          loss = 'categorical_crossentropy',
          optimizer = 'Adam',metrics = ['accuracy']
                )
    return model 


model = make_model()
model.summary()


h = model.fit(
      train_set,validation_data = validation_set,
                              epochs=20,steps_per_epoch = 64,validation_steps = 48,
                              callbacks = [
                              keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,mode='auto'),
                              keras.callbacks.ModelCheckpoint('explo/model_{val_loss:.3f}.h5',
                              save_best_only = True,save_weights_only=False,
                              monitor='val_loss')
                              ]


)
model.save('explo_model.h5')


## Loading the saved model
from keras.models import load_model
model = load_model('explo_model.h5')

%matplotlib inline
accu= h.history['accuracy']
val_acc=h.history['val_accuracy']
loss=h.history['loss']
val_loss=h.history['val_loss']

epochs=range(len(accu)) #No. of epochs

import matplotlib.pyplot as plt
plt.plot(epochs,accu,'r',label='Training Accuracy')
plt.plot(epochs,val_acc,'g',label='Testing Accuracy')
plt.legend()
plt.xlabel('No. of epochs')
plt.ylabel('Accuracy score')
plt.figure()

#Plot training and validation loss per epoch
plt.plot(epochs,loss,'r',label='Training Loss')
plt.plot(epochs,val_loss,'g',label='Testing Loss')
plt.xlabel('No. of epochs')
plt.ylabel('Loss score')
plt.legend()
plt.show()
		
train_set.class_indices

labels = ['0','1','+','-','*','/','Confirm','**','%','Clear','2','3','4','5','6','7','8','9']

