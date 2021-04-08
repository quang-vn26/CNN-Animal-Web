import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import cv2
from PIL import Image
from glob import glob
import sklearn
import pickle

haar = cv2.CascadeClassifier('./module/haarcascade_frontalface_default.xml')
mean = pickle.load(open('./module/mean_preprocess.pickle','rb'))
model_svm = pickle.load(open('./module/model_svm.pickle','rb'))
model_pca = pickle.load(open('./module/pca_50.pickle','rb')) 
print("module load success")

#setting
gender_pre = ['Male','Female']
font = cv2.FONT_HERSHEY_SIMPLEX

def pipeline_model(path,filename,color = 'bgr'):
    #step 1 read image
    img = cv2.imread(path)
    #step 2 convert into gray scale
    if color == 'bgr':
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #step 3 crop the face (using haar cascade classifier)
    faces = haar.detectMultiScale(gray,1.5,3)#?
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
        roi = gray[y:y+h,x:x+w]#crop image
        #step 4 normalization(0-1)
        roi = roi/255.0
        #step 5 resize image 100x100
        if roi.shape[1] >100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
        #step 6 Flattening (1x10000)
        roi_reshape = roi_resize.reshape(1,10000)#1,-1
        #step 7: subptract with mean
        roi_mean = roi_reshape - mean
        #step 8: get eigen image
        eigen_image = model_pca.transform(roi_mean)
        #step 9: pass to ml model (svm)
        results = model_svm.predict_proba(eigen_image)[0]
        #step 10: 
        predict = results.argmax()# 0  or 1
        score = results[predict]
        #step 11:
        text = "%s : %0.2f"%(gender_pre[predict],score)
        cv2.putText(img,text,(x,y),font,1.0,(225,255,0),2)

    cv2.imwrite('./static/predict/{}'.format(filename),img)    
