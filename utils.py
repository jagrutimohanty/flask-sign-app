import os 
import numpy as np
import cv2 as cv2
import math
import os
import time
import re
import shutil
from six.moves import urllib

from PIL import Image
from io import BytesIO
import sys
import ast

import tensorflow as tf


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_chunks(l, n):
  """Yield successive n-sized chunks from l.
  Used to create n sublists from a list l"""
  for i in range(0, len(l), n):
    yield l[i:i + n]


def get_video_capture_and_frame_count(path):
  assert os.path.isfile(
    path), "Couldn't find video file:" + path + ". Skipping video."
  cap = None
  if path:
    cap = cv2.VideoCapture(path)

  assert cap is not None, "Couldn't load video capture:" + path + ". Skipping video."

  # compute meta data of video
  if hasattr(cv2, 'cv'):
    frame_count = int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
  else:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  return cap, frame_count


def get_next_frame(cap):
  ret, frame = cap.read()
  if not ret:
    return None

  return np.asarray(frame)


def compute_dense_optical_flow(prev_image, current_image):
  old_shape = current_image.shape
  prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
  current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
  assert current_image.shape == old_shape
  hsv = np.zeros_like(prev_image)
  hsv[..., 1] = 255
  flow = None
  flow = cv2.calcOpticalFlowFarneback(prev=prev_image_gray,
                                      next=current_image_gray, flow=flow,
                                      pyr_scale=0.8, levels=15, winsize=5,
                                      iterations=10, poly_n=5, poly_sigma=0,
                                      flags=10)

  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  hsv[..., 0] = ang * 180 / np.pi / 2
  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def video_file_to_ndarray(i, file_path, n_frames_per_video, height, width,
                          n_channels, num_real_image_channel,
                          dense_optical_flow, number_of_videos):
    cap, frame_count = get_video_capture_and_frame_count(file_path)

    take_all_frames = False
    # if not all frames are to be used, we have to skip some -> set step size accordingly
    if n_frames_per_video == 'all':
        take_all_frames = True
        video = np.zeros((frame_count, height, width, n_channels), dtype=np.uint32)
        steps = frame_count
        n_frames = frame_count
    else:
        video = np.zeros((n_frames_per_video, height, width, n_channels),
                     dtype=np.uint32)
        steps = int(math.floor(frame_count / n_frames_per_video))
        n_frames = n_frames_per_video

    assert not (frame_count < 1 or steps < 1), str(
      file_path) + " does not have enough frames. Skipping video."

    # variables needed
    image = np.zeros((height, width, num_real_image_channel),
                    dtype=np.uint8)
    frames_counter = 0
    prev_frame_none = False
    restart = True
    image_prev = None

    while restart:
        for f in range(frame_count):
            if math.floor(f % steps) == 0 or take_all_frames:
                frame = get_next_frame(cap)
                # unfortunately opencv uses bgr color format as default
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # special case handling: opencv's frame count sometimes differs from real frame count -> repeat
                if frame is None and frames_counter < n_frames:
                    stop, _, _1, _2, _3, _4 = repeat_image_retrieval(
                    cap, file_path, video, take_all_frames, steps, frame, prev_frame_none,
                      frames_counter)
                    if stop:
                        restart = False
                        break
                    else:
                        video[frames_counter, :, :, :].fill(0)
                        frames_counter += 1

            else:
                if frames_counter >= n_frames:
                    restart = False
                    break

                # iterate over channels
                for k in range(num_real_image_channel):
                    resizedImage = cv2.resize(frame[:, :, k], (width, height))
                    image[:, :, k] = resizedImage

                if dense_optical_flow:
                    # optical flow requires at least two images, make the OF image appended to the first image just black
                    if image_prev is not None:
                        frame_flow = compute_dense_optical_flow(image_prev, image)
                        frame_flow = cv2.cvtColor(frame_flow, cv2.COLOR_BGR2GRAY)
                    else:
                        frame_flow = np.zeros((height, width))
                    image_prev = image.copy()

                # assemble the video from the single images
                if dense_optical_flow:
                    image_with_flow = image.copy()
                    image_with_flow = np.concatenate(
                      (image_with_flow, np.expand_dims(frame_flow, axis=2)), axis=2)
                    video[frames_counter, :, :, :] = image_with_flow
                else:
                    video[frames_counter, :, :, :] = image
                frames_counter += 1
        else:
            get_next_frame(cap)

    print(str(i + 1) + " of " + str(
      number_of_videos) + " videos within batch processed: ", file_path)

    v = video.copy()
    cap.release()
    return v

def convert_video_to_numpy(filenames, n_frames_per_video, width, height,
                           n_channels, dense_optical_flow=False):
    """Generates an ndarray from multiple video files given by filenames.
  Implementation chooses frame step size automatically for a equal separation distribution of the video images.

  Args:
    filenames: a list containing the full paths to the video files
    width: width of the video(s)
    height: height of the video(s)
    n_frames_per_video: integer value of string. Specifies the number of frames extracted from each video. If set to 'all', all frames are extracted from the
    videos and stored in the tfrecord. If the number is lower than the number of available frames, the subset of extracted frames will be selected equally
    spaced over the entire video playtime.
    n_channels: number of channels to be used for the tfrecords
    type: processing type for video data

  Returns:
    if no optical flow is used: ndarray(uint32) of shape (v,i,h,w,c) with
    v=number of videos, i=number of images, (h,w)=height and width of image,
    c=channel, if optical flow is used: ndarray(uint32) of (v,i,h,w,
    c+1)
  """

    number_of_videos = len(filenames)

    if dense_optical_flow:
        # need an additional channel for the optical flow with one exception:
        n_channels = 4
        num_real_image_channel = 3
    else:
        # if no optical flow, make everything normal:
        num_real_image_channel = n_channels

    data = []
    name_to_append=[]

    for i, file in enumerate(filenames):
        try:
            v = video_file_to_ndarray(i=i, file_path=file,
                                    n_frames_per_video=n_frames_per_video,
                                    height=height, width=width,
                                    n_channels=n_channels,
                                    num_real_image_channel=num_real_image_channel,
                                    dense_optical_flow=dense_optical_flow,
                                    number_of_videos=number_of_videos)
#             print(v)
            name_to_append.append(file)
            data.append(v)
        except Exception as e:
            print(e)

    return np.array(data),np.array(name_to_append)


def convert_video_to_tensor(filename):
  arr = convert_video_to_numpy(
        filenames = [filename], 
        n_frames_per_video = 5, 
        width = 224, 
        height = 224,
        n_channels = 3,
        dense_optical_flow=False)[0]

  arr = np.true_divide(arr, 255)

  return tf.reshape(
    tensor=tf.convert_to_tensor(arr, dtype=tf.float32),
    shape=(1, 5, 224, 224, 3)
  )