# things we need for Tensorflow
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

model = tf.keras.models.load_model('model_save_cnn_nomal_1.h5')
model.summary()
categories = animals =['butterfly', 'chicken', 'elephant', 'horse', 'spider', 'squirel']

HEIGHT = 32
WIDTH = 55

path_img = 'b.jpeg'

def classify(path_img):
    test_data=[]
    test_image_o = cv2.imread(path_img)
    test_image = cv2.resize(test_image_o, (WIDTH, HEIGHT))  # .flatten()
    test_data.append(test_image)
    # scale the raw pixel intensities to the range [0, 1]
    test_data = np.array(test_image, dtype="float") / 255.0
    test_data=test_data.reshape([-1,32, 55, 3])
    pred = model.predict(test_data)
    predictions = argmax(pred, axis=1) # return to label
    animal_label=categories[predictions[0]]
    print ('Prediction : '+animal_label)
    
    for idx, animal, x in zip(range(0,len(categories)), animals , pred[0]):
        str_animal = "ID: "+str(idx)+", Label:"+str(animal)+" "+str(round(x*100,2))+"%"
        # print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2) ))
        arr_y = []
        arr_y.append(str_animal)
        print(arr_y)

        # tmp = "ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2)
    return animal_label

if __name__ == '__main__':
    print('Begin Testing: ')
    result = classify(path_img)
    print(result)


	# while True:
	# 	print('Human: ', end=''),
	# 	x = input()
	# 	tag, _ = classify(x)
	# 	print('Bot: ', response(tag))

	# 	# tag, conf = classify(x)
	# 	# print('Bot: ', tag, conf)

