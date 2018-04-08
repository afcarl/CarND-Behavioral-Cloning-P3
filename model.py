# utility packages
import os
from tqdm import tqdm
import csv

# math and support packages
from scipy import ndimage
import numpy as np
import pandas as pd

# image processing packages
import cv2
import matplotlib.pyplot as plt

img_data = []
IMG_PATH = "IMG"
# All images are stored in a directory called IMG living along side model.py path

def normalize(img):
# image data should be normalized so that the data has mean zero and equal variance.
    return (img - 128.) / 128.

def grayscale(img):
# remove color channels. image shape should now be (320,160,1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def preprocess(img):
    # this is our main preprocess pipeline

    img1 = grayscale(img)
    img1.reshape((160,320,1))
    # grayscale has to come first for cv2
    # error: (-215) depth == CV_8U || depth == CV_16U || depth == CV_32F
    # in function cvtColor

    img2 = normalize(img1)

    # TODO: Cut image sizes, add filter polygons
    # img3 = filter_images(img2)

    return img2

def load_images():
    image_directory = os.path.join(os.getcwd(), IMG_PATH)
    lines = list()

    with open('driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    image_data = list()

    for row in tqdm(lines):
        # center_image = cv2.imread(row['center_camera_fp'])
        # left_image = cv2.imread(row['left_camera_fp'])
        # right_image = ndimage.imread(row['right_camera_fp'])
        center_image = cv2.imread(row[0])
        left_image = cv2.imread(row[1])
        right_image = cv2.imread(row[2])
        # each of these should contain an nd array

        data = dict()
        data['X'] = {'center_image': center_image,
                     'left_image': left_image,
                     'right_image': right_image}
        data['y'] = {'steering_angle': row[3],
                     'throttle': row[4],
                     'brake': row[5],
                     'speed': row[6]}

        image_data.append(data)

    return image_data
    # returns a list of dicts, each dict containing pointers to the ndarrays
    # for each of the 3 camera images, and labels for steering, braking

def process_pipeline(img_data):
    # load images as X
    # for data in tqdm(img_data, desc='process_pipeline', unit='images'):
    #     data['X']['center_image'] = preprocess(data['X']['center_image'])
    #     data['X']['left_image'] = preprocess(data['X']['left_image'])
    #     data['X']['right_image'] = preprocess(data['X']['right_image'])

    for i, img in enumerate(img_data):
        img_data[i] = preprocess(img)
    return img_data

def model_basic():
# define models

    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))

    return model

def model_lenet():
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda
    from keras.layers.convolutional import Convolution2D, MaxPooling2D

    model = Sequential()
    model.add(Convolution2D(6,5,5, activation='relu', input_shape=(160,320,3)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def _retrieve_center_image(data):
    return data['X']['center_image']

def retrieve_images_and_labels(img_data):
    images = []
    measurements = []
    for img in img_data:
        center_img_data = _retrieve_center_image(data=img)
        measurement = _retrieve_steering_angle(data=img)

        images.append(center_img_data)
        measurements.append(measurement)

        image_flipped = np.fliplr(center_img_data)
        measurement_flipped = -measurement

        images.append(image_flipped)
        measurements.append(measurement_flipped)

    return images, measurements

def _retrieve_steering_angle(data):
    return float(data['y']['steering_angle'])


img_data = load_images()

# img_data = process_pipeline(img_data)
# WE can't do any kind of grayscale preprocessing since the simulator won't
# feed those images in to the model
images, measurements = retrieve_images_and_labels(img_data)


X = np.array(images)
y = np.array(measurements)
# model = model_basic()
model = model_lenet()
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')


# save model to h5
