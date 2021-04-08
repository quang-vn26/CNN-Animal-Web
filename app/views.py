from __future__ import absolute_import
from flask import render_template,request,redirect,url_for
import os
from PIL import Image
# from utils import pipeline_model
from .utils import pipeline_model


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
def faceapp():
    if request.method == 'POST':
        f = request.files['image']
        filename = f.filename
        path = os.path.join(upload_path,filename)
        f.save(path)
        w = getwidth(path)
        #predict
        pipeline_model(path,filename,color='bgr')
        return render_template('faceapp.html',fileupload=True,img_name=filename,w = w)        
    return render_template('faceapp.html',fileupload=False,img_name="home.svg",w="300")    