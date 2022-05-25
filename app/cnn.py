import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
import numpy
import pickle
import json
import cv2
from numpy import argmax

model = tf.keras.models.load_model('module/model_save_cnn_nomal_1.h5')
# model.summary()
categories =animals=['butterfly', 'chicken', 'elephant', 'horse', 'spider', 'squirel']
HEIGHT = 32
WIDTH = 55

def classify(path,filename):
    path_img = path
    test_data=[]
    test_image_o = cv2.imread(path_img)
    test_image = cv2.resize(test_image_o, (WIDTH, HEIGHT))  # .flatten()
    test_data.append(test_image)
    # scale the raw pixel intensities to the range [0, 1]
    test_data = np.array(test_image, dtype="float") / 255.0
    test_data=test_data.reshape([-1,32, 55, 3])
    pred = model.predict(test_data)
    predictions = argmax(pred, axis=1) 
    animal_label=categories[predictions[0]]
    print ('Prediction : '+animal_label)
    score = 0
    for idx, animal, x in zip(range(0,len(categories)), animals , pred[0]):
        str_animal = "ID: "+str(idx)+", Label:"+str(animal)+" "+str(round(x*100,2))+"%"
        tmp = round(x*100,2)
        if score<tmp:
            score = tmp
        arr_y = []
        arr_y.append(str_animal)
        # print(arr_y)

    return animal_label,score

