from __future__ import absolute_import
from flask import render_template,request,redirect,url_for
import os
from PIL import Image
from .cnn import classify
from .vgg16_tf import test_single_image

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

upload_path = 'static/uploads' 
def index():
    return render_template('index.html')
def base():
    return render_template('base.html')    
def getwidth(path):    
    img = Image.open(path)
    size = img.size
    aspect = size[0]/size[1]
    w = 300*aspect
    return int(w)
def cnn():
    if request.method == 'POST':
        f = request.files['image']
        filename = f.filename
        path = os.path.join(upload_path,filename)
        f.save(path)
        w = getwidth(path)
        #predict
        result_predit,score = classify(path,filename)
        # result_predit = 'ket qua'
        return render_template('faceapp.html',fileupload=True,img_name=filename,w = w,result_predit = result_predit,score=score)        
    return render_template('faceapp.html',fileupload=False,img_name="home.svg",w="300")   

def vgg16_tf():
    if request.method == 'POST':
        f = request.files['image']
        filename = f.filename
        path = os.path.join(upload_path,filename)
        f.save(path)
        w = getwidth(path)
        #predict
        result_predit,score = test_single_image(path)

        return render_template('vgg16.html',fileupload=True,img_name=filename,w = w,result_predit = result_predit,score=score)        
    return render_template('vgg16.html',fileupload=False,img_name="home.svg",w="300")   
