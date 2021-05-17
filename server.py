from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
from tensorflow.python.platform import gfile
import tensorflow as tf 
import os
import json
import time
import sys
from multiprocessing.dummy import Pool
from tensorflow.python.platform import flags
import cv2 as cv2
import numpy as np
import math
import re
from tensorflow.python.platform import app
import tensorflow_hub as hub
from utils import convert_video_to_tensor

app = Flask(__name__)
 
video_camera = None
global_frame = None
model = None
predict_func = None
LABELS = ['beautiful', 'hello', 'please', 'sorry']



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")

def video_stream():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()
        
    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



def preprocesspredict():
    test_datagen = ImageDataGenerator(rescale = 1./255)
    vals = ['Cat', 'Dog'] # change this to the labels to predict
    test_dir = './static/video.mp4'
    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size =(224, 224),
            color_mode ="rgb",
            shuffle = False,
            class_mode ='categorical',
            batch_size = 1)
    pred = model.predict_generator(test_generator)
    print(pred)
    return str(vals[np.argmax(pred)])




def predfunc():
    p = [0.5,0.7,.09,.7,.1,.2,.3,.5]

@app.route('/predict', methods=['GET','POST'])
def predict():
    global model
    global predict_func
    global LABELS

    if not model or not predict_func:
        model = tf.saved_model.load('./static/trained_models/2')
        predict_func = model.signatures["serving_default"]

    pred = predict_func(convert_video_to_tensor('./static/video.mp4'))['output_0']
    layer = tf.keras.layers.Softmax()
    p = layer(pred).numpy()

    outputdict = {}
    
    for idx, label in enumerate(LABELS):
        outputdict[label] = p[0][idx] 

    label = LABELS[np.argmax(p[0])]

    return render_template('pred.html', res=outputdict, label=label)  



if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
