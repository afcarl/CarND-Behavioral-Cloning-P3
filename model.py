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
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

img_data = []
IMG_PATH = "IMG"
# All images are stored in a directory called IMG living along side model.py path

def normalize(img):
# image data should be normalized so that the data has mean zero and equal variance.
    return (img - 128.) / 128.

def grayscale(img):
# remove color channels. image shape should now be (320,160,1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def blur(img):
    return cv2.blur(img, (6,6))

def crop_img(img):
    start_of_landscape = 60
    length_of_windshield = 70
    cropped_y = start_of_landscape + length_of_windshield
    cropped_img = img[start_of_landscape:cropped_y, 0:320]
    return cropped_img

def preprocess(img):
    # this is our main preprocess pipeline

    # img1 = grayscale(img)
    # img1.reshape((160,320,1))
    img1 = img
    # grayscale has to come first for cv2
    # error: (-215) depth == CV_8U || depth == CV_16U || depth == CV_32F
    # in function cvtColor

    img2 = normalize(img1)

    # TODO: Cut image sizes, add filter polygons
    # img3 = filter_images(img2)
    img3 = blur(img2)

    # img4 = crop_img(img3)

    return img3

def _get_img_path(orig_path):
    jpg_file_path = 'IMG/' +  orig_path.split('IMG')[1]
    center_image_fp = os.path.join(os.getcwd(), jpg_file_path)
    return center_image_fp

def load_images_generator(batch_size=32):
    image_directory = os.path.join(os.getcwd(), IMG_PATH)

    i = 0
    measurements = []
    images = []
    with open('driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if i == batch_size:
                images = np.array(images)
                measurements = np.array(measurements)
                images, measurements = sklearn.utils.shuffle(images, measurements)
                yield (images, measurements)
                images = []
                measurements = []
                i = 0

            center_image_fp = _get_img_path(row[0])
            center_image = cv2.imread(center_image_fp)

            center_image_fp = _get_img_path(row[1])
            left_image = cv2.imread(center_image_fp)

            center_image_fp = _get_img_path(row[2])
            right_image = cv2.imread(center_image_fp)

            # each of these should contain an nd array

            data = dict()
            data['X'] = {'center_image': center_image,
                         'left_image': left_image,
                         'right_image': right_image}
            data['y'] = {'steering_angle': row[3],
                         'throttle': row[4],
                         'brake': row[5],
                         'speed': row[6]}

            # image_data.append(data)
            center_img_data = _retrieve_center_image(data=data)
            measurement = _retrieve_steering_angle(data=data)
            center_img_data = process_pipeline(center_img_data)

            images.append(center_img_data)
            measurements.append(measurement)

            # image_flipped = np.fliplr(center_img_data)
            # measurement_flipped = -measurement
            #
            # # yield (np.array(center_img_data), np.array(measurement))
            #
            # images.append(image_flipped)
            # measurements.append(measurement_flipped)

def load_images():
    image_directory = os.path.join(os.getcwd(), IMG_PATH)

    lines = list()
    i = 0

    with open('driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # lines.append(line)

            row = line
            # for row in tqdm(lines):
            # center_image = cv2.imread(row['center_camera_fp'])
            # left_image = cv2.imread(row['left_camera_fp'])
            # right_image = ndimage.imread(row['right_camera_fp'])

            center_image_fp = _get_img_path(row[0])
            center_image = cv2.imread(center_image_fp)

            center_image_fp = _get_img_path(row[1])
            left_image = cv2.imread(center_image_fp)

            center_image_fp = _get_img_path(row[2])
            right_image = cv2.imread(center_image_fp)

            # each of these should contain an nd array

            data = dict()
            data['X'] = {'center_image': center_image,
                         'left_image': left_image,
                         'right_image': right_image}
            data['y'] = {'steering_angle': row[3],
                         'throttle': row[4],
                         'brake': row[5],
                         'speed': row[6]}

            # image_data.append(data)
            yield data


    # return image_data
    # returns a list of dicts, each dict containing pointers to the ndarrays
    # for each of the 3 camera images, and labels for steering, braking

def process_pipeline(img_data):
    # load images as X
    # for data in tqdm(img_data, desc='process_pipeline', unit='images'):
    #     data['X']['center_image'] = preprocess(data['X']['center_image'])
    #     data['X']['left_image'] = preprocess(data['X']['left_image'])
    #     data['X']['right_image'] = preprocess(data['X']['right_image'])

    return preprocess(img_data)

def model_basic(input_shape=None):
# define models

    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(70,320,3)))
    model.add(Flatten())
    model.add(Dense(1))

    return model

def model_lenet(input_shape=None):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda
    from keras.layers.convolutional import Convolution2D, MaxPooling2D

    model = Sequential()
    model.add(Convolution2D(6,5,5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def model_nvidia(input_shape=None):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Dropout, BatchNormalization, Activation
    from keras.layers import Flatten, Dense, Lambda, Cropping2D

    # activation = "relu"
    # model = Sequential()
    #
    # # Normalize
    # # model.add(BatchNormalization(input_shape=input_shape, axis=1))
    #
    # model.add(Convolution2D(24, 5, 5, activation=activation, input_shape=input_shape, name="convolution0"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #
    # model.add(Convolution2D(36, 5, 5,  activation=activation, name="convolution1"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #
    # model.add(Convolution2D(48, 5, 5, activation=activation, name="convolution2"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #
    # model.add(Convolution2D(64, 3, 3, activation=activation, name="convolution3"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #
    # model.add(Convolution2D(64, 3, 3, activation=activation, name="convolution4"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #
    #
    # model.add(Flatten())
    # # model.add(Dropout(0.2))
    # # model.add(Activation(activation=activation))
    #
    # model.add(Dense(1164, activation=activation, name="dense1"))
    # # model.add(Dropout(0.5))
    # # model.add(Activation(activation=activation))
    #
    # model.add(Dense(100, activation=activation, name="dense2"))
    # # model.add(Dropout(0.5))
    # # model.add(Activation(activation=activation))
    #
    # model.add(Dense(50, activation=activation, name="dense3"))
    # # model.add(Dropout(0.5))
    # # model.add(Activation(activation=activation))
    #
    # model.add(Dense(10, activation=activation, name="dense4"))
    # # model.add(Dropout(0.5))
    # # model.add(Activation(activation=activation))
    # model.add(Dense(1, name="output"))

    activation = 'relu'

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24, 5, 5, activation=activation, input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Convolution2D(36, 5, 5, activation=activation))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48, 5, 5, activation=activation))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3,3, activation=activation))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 1, 1, activation=activation))
    model.add(MaxPooling2D())
    model.add(Flatten())
    # model.add(Dense(1164))
    # model.add(Dropout(0.5))

    model.add(Dense(100))
    # model.add(Dropout(0.5))

    model.add(Dense(50))
    # model.add(Dropout(0.5))

    model.add(Dense(10))
    # model.add(Dropout(0.5))

    model.add(Dense(1))
    return model


def _retrieve_center_image(data):
    return data['X']['center_image']

def retrieve_images_and_labels(batch_size=32):

    images = []
    measurements = []
    i = 0
    for img in load_images():
        if i == batch_size:
            images = np.array(images)
            measurements = np.array(measurements)
            images, measurements = sklearn.utils.shuffle(images, measurements)
            yield (images, measurements)
            images = []
            measurements = []
            i = 0

        center_img_data = _retrieve_center_image(data=img)
        measurement = _retrieve_steering_angle(data=img)
        center_img_data = process_pipeline(center_img_data)

        images.append(center_img_data)
        measurements.append(measurement)

        image_flipped = np.fliplr(center_img_data)
        measurement_flipped = -measurement

        # yield (np.array(center_img_data), np.array(measurement))

        images.append(image_flipped)
        measurements.append(measurement_flipped)
        # yield (np.array(image_flipped), np.array(measurement_flipped))
        i += 1



    # return images, measurements

def _retrieve_steering_angle(data):
    return float(data['y']['steering_angle'])

def get_number_of_samples():
    data_dir_path = os.path.join( os.getcwd(), IMG_PATH)
    return len([name for name in os.listdir(data_dir_path)])


# img_data = process_pipeline(img_data)
# WE can't do any kind of grayscale preprocessing since the simulator won't
# feed those images in to the model
# images, measurements = retrieve_images_and_labels()
#
# X = np.array(images)
# y = np.array(measurements)
#
# X = np.array([images[0]])
# y = np.array([measurements[0]])
#
# for image in images:
#     image = np.array([image])
#     X = np.concatenate( (X, image) )
#
# for measurement in measurements:
#     measurement = np.array([measurement])
#     y = np.concatenate( (y, measurement) )

# model = model_basic()

def get_samples():
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

samples = get_samples()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                center_image = preprocess(center_image)
                images.append(center_image)
                angles.append(center_angle)

                name = './IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = float(batch_sample[3]) + 0.2
                left_image = preprocess(left_image)
                images.append(left_image)
                angles.append(left_angle)

                name = './IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = float(batch_sample[3]) - 0.2
                right_image = preprocess(right_image)
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = model_nvidia(input_shape=(160,320,3))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3,
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples)*3, nb_epoch=4)
# model.compile(loss='mse', optimizer='adam')
# model.fit_generator(generator=load_images_generator(), samples_per_epoch=number_of_samples,
#     # validation_split=0.2, shuffle=True,
#     nb_epoch=2)

model.save('model.h5')


# save model to h5
