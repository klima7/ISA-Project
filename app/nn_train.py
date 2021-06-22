import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Disable tensorflow logging

import math
import random
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.layers import InputLayer, Conv2D, Flatten, Dense
from app.constants import SignType, TrafficLightColor


signs_model = keras.models.Sequential([
    InputLayer(input_shape=(16, 16, 1)),
    Conv2D(4, (3, 3), activation='relu'),
    Flatten(),
    Dense(150, activation='relu'),
    Dense(70, activation='relu'),
    Dense(9, activation='softmax')
])


lights_model = keras.models.Sequential([
    InputLayer(input_shape=(16, 16, 1)),
    Conv2D(4, (3, 3), activation='relu'),
    Flatten(),
    Dense(150, activation='relu'),
    Dense(70, activation='relu'),
    Dense(4, activation='softmax')
])


def load_images_from_dir(directory, max_count=math.inf):
    names = []

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            names.append(filename)

    images = []
    count = 0

    while names:
        if count >= max_count:
            break
        name = random.choice(names)
        names.remove(name)
        image = cv2.imread(os.path.join(directory, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255
        images.append(image)
        count += 1

    return images


def load_signs_training_data(directory):
    rev_count = 100

    stop = load_images_from_dir(os.path.join(directory, 'stop', 'normal'), max_count=600)
    walkway = load_images_from_dir(os.path.join(directory, 'walkway', 'normal'))
    roundabout = load_images_from_dir(os.path.join(directory, 'roundabout', 'normal'))
    parking = load_images_from_dir(os.path.join(directory, 'parking', 'normal'))
    limit = load_images_from_dir(os.path.join(directory, 'limit', 'normal'))
    oneway = load_images_from_dir(os.path.join(directory, 'oneway', 'normal'))
    deadend = load_images_from_dir(os.path.join(directory, 'deadend', 'normal'))
    lights = load_images_from_dir(os.path.join(directory, 'lights', 'all'))

    reversed = [
        *load_images_from_dir(os.path.join(directory, 'stop', 'reversed'), max_count=rev_count),
        *load_images_from_dir(os.path.join(directory, 'walkway', 'reversed'), max_count=rev_count),
        *load_images_from_dir(os.path.join(directory, 'roundabout', 'reversed'), max_count=rev_count),
        *load_images_from_dir(os.path.join(directory, 'parking', 'reversed'), max_count=rev_count),
        *load_images_from_dir(os.path.join(directory, 'limit', 'reversed'), max_count=rev_count),
        *load_images_from_dir(os.path.join(directory, 'oneway', 'reversed'), max_count=rev_count),
        *load_images_from_dir(os.path.join(directory, 'deadend', 'reversed'), max_count=rev_count),
    ]

    data = {
        SignType.STOP: stop,
        SignType.WALKWAY: walkway,
        SignType.ROUNDABOUT: roundabout,
        SignType.PARKING: parking,
        SignType.LIMIT: limit,
        SignType.ONEWAY: oneway,
        SignType.DEADEND: deadend,
        SignType.TRAFFIC_LIGHTS: lights,
        SignType.REVERSED: reversed
    }

    return data


def load_lights_training_data(lights_directory):
    red = load_images_from_dir(os.path.join(lights_directory, 'red'))
    yellow = load_images_from_dir(os.path.join(lights_directory, 'yellow'))
    green = load_images_from_dir(os.path.join(lights_directory, 'green'))
    none = load_images_from_dir(os.path.join(lights_directory, 'none'))

    data = {
        TrafficLightColor.RED: red,
        TrafficLightColor.YELLOW: yellow,
        TrafficLightColor.GREEN: green,
        TrafficLightColor.NONE: none,
    }

    return data


def prepare_training_data(data, onehot_max):
    rows = []

    for label, images in data.items():
        for image in images:
            row = (image, label)
            rows.append(row)

    random.shuffle(rows)

    images = np.array([row[0] for row in rows])
    labels = np.array([row[1].value for row in rows])
    labels = tf.one_hot(labels, onehot_max)

    images = np.reshape(images, (images.shape[0], 16, 16, 1))
    return images, labels


def train_signs_nn():
    data = load_signs_training_data('../images/gray16')
    images, labels = prepare_training_data(data, 9)

    signs_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    signs_model.fit(images, labels, batch_size=16, epochs=20, verbose=1)
    signs_model.save('../nn_signs.h5')


def train_lights_nn():
    data = load_lights_training_data('../images/gray16/lights')
    images, labels = prepare_training_data(data, 4)

    lights_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lights_model.fit(images, labels, batch_size=16, epochs=20, verbose=1)
    lights_model.save('../nn_lights.h5')


if __name__ == '__main__':
    # train_signs_nn()
    train_lights_nn()
