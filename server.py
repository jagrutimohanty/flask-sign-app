from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
from tensorflow.python.platform import gfile
import tensorflow as tf 
import os
import json
import time
import sys
#import urllib.request
from multiprocessing.dummy import Pool
from tensorflow.python.platform import flags
import cv2 as cv2
import numpy as np
import math
import re
from tensorflow.python.compiler.mlcompute import mlcompute



app = Flask(__name__)

video_camera = None
global_frame = None


###############



#########################


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
    proba = [0.5,0.7,.09,.7,.1,.2,.3,.5]
    ##loading the model from the saved file
    # model = tf.keras.models.load_model('/Users/jagrutimohanty/flask-sign-app-may15/static/trained_models/2/')
    # print(model.summary())
  #  return proba

@app.route('/predict', methods=['GET','POST'])
def predict():
  #  test_datagen = ImageDataGenerator(rescale = 1./255)
    vals = ['beautiful', 'hello', 'please', 'sorry'] 
  #  ['beautiful', 'hello', 'please', 'sorry'] # change this to the labels to predict
    #test_dir = './static/video.mp4'
    model = tf.keras.models.load_model('/Users/jagrutimohanty/flask-sign-app-may15/static/trained_models/2')
    print(model.summary())
    print("Jagruti")
    proba = [0.5,0.7,.09,.7]
    predfunc()
    top_3 = np.argsort(proba)[:-4:-1]
    print(top_3)
    #probs_val = proba[top_3]
    #prob_lab = vals[top_3]
    probs_lab = []
    probs_val = []
    for i in range (len(top_3)):
      probs_val.append(proba[top_3[i]])
      probs_lab.append(vals[top_3[i]])
    #lab = str(vals[top_3])
    #outputdict = {'a': 1, 'b': 2, 'c': 3}
    outputdict = dict(zip(probs_lab, probs_val))
    print(outputdict)
    #str(vals[np.argmax(pred)])
    return render_template('pred.html', res = outputdict )  







if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)


#refered link https://www.geeksforgeeks.org/deploying-a-tensorflow-2-1-cnn-model-on-the-web-with-flask/