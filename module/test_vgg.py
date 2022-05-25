import pandas as pd
import numpy as np 
import keras
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense ,LeakyReLU
from keras import applications  
import math  
from tensorflow.keras import optimizers
# import tensorflow.keras.activations.LeakyReLU
import time
from keras.utils.np_utils import to_categorical  


path_img = '5.jpg'

def read_image(file_path):
    image = load_img(file_path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image

def test_single_image(path):
    vgg16 = applications.VGG16(include_top=False, weights='imagenet') 
    # model = keras.models.load_model('model_vgg_2.h5')
    model = keras.models.load_model('model_vgg_2.h5',  custom_objects={'LeakyReLU': keras.layers.LeakyReLU})

    animals = ['butterflies', 'chickens', 'elephants', 'horses', 'spiders', 'squirells']
    images = read_image(path)
    time.sleep(.5)
    bt_prediction = vgg16.predict(images)  
    preds = model.predict(bt_prediction)
    for idx, animal, x in zip(range(0,6), animals , preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2) ))
    print('Final Decision:')
    time.sleep(.5)
    for x in range(3):
        print('.'*(x+1))
        time.sleep(.2)
    classes_x=np.argmax(preds,axis=1)    
    class_predicted = classes_x
    return animals[class_predicted[0]]

if __name__ == '__main__':
    print('Begin Testing: ')
    result = test_single_image(path_img)
    print(result)
