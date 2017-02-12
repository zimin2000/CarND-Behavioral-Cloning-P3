import sklearn
import cv2
import os
import csv
import pickle
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout, Activation
from keras.models import Model, Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

DATA_DIR = "./data"
BATCH_SIZE = 32

IMG_WIDTH = 320
IMG_HEIGHT = 160

CROP_TOP = 50
CROP_BOTTOM = 20
CROP_SIDE = 1

samples = []
with open(os.path.join(DATA_DIR, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] == 'center':
            continue

        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)

    while True: # Loop forever so the generator never terminates
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for sample in batch_samples:
                name = os.path.join(DATA_DIR, 'IMG', sample[0].split('/')[-1])

                center_image = cv2.imread(name)

                center_angle = float(sample[3])

#                print "{} {} {}".format(name, center_image.shape, center_angle)

                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield (X_train, y_train)

def train():
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size = BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)

    model = Sequential()

    # Preprocessing 0.1: Cropping useless top and bottom pixels
    # 320 x 160 x 3 -> 3 x 320 x 90
    model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (CROP_SIDE, CROP_SIDE)), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), dim_ordering='tf'))

    # Preprocessing 0.2: Centered around zero with small standard deviation
    # 320 x 90 x 3 -> 320 x 90 x 3
#    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # Layer 1:
    # 3 x 318 x 90 -> Conv 7 x 7 (+1 x +1) -> MaxPool 3 x 3 -> Dropout -> Relu -> 24 x 104 x 28
    model.add(Convolution2D(24, 7, 7, border_mode='valid'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Dropout(0.95))
    model.add(Activation('relu'))


    # Layer 2:
    # 24 x 104 x 28 -> Conv 5 x 5 (+1 x +1) -> MaxPool 2 x 2 -> Dropout -> Relu -> 36 x 50 x 12
    model.add(Convolution2D(36, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.95))
    model.add(Activation('relu'))


    # Layer 3:
    # 36 x 50 x 12 -> Conv 5 x 5 (+1 x +1) -> MaxPool 2 x 2 -> Dropout -> Relu -> 48 x 23 x 4
    model.add(Convolution2D(48, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.95))
    model.add(Activation('relu'))


    # Layer 4:
    # 48 x 23 x 4 -> Conv 3 x 3 -> Dropout -> Relu -> 64 x 21 x 2
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Dropout(0.95))
    model.add(Activation('relu'))


    # Layer 5:
    # 64 x 21 x 2 -> Conv 3 x 2 -> Dropout -> Relu -> 64 x 19 x 1
    model.add(Convolution2D(64, 2, 3, border_mode='valid'))
    model.add(Dropout(0.95))
    model.add(Activation('relu'))

    # FC Layer 1:
    # 64 x 19 x 1 -> Dense -> 1164
    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Dropout(0.95))
    model.add(Activation('relu'))
    
    # FC Layer 2:
    # 1164 -> Dense -> 100
    model.add(Dense(100))
    model.add(Dropout(0.95))
    model.add(Activation('relu'))
    
    # FC Layer 3:
    # 100 -> Dense -> 50
    model.add(Dense(50))
    model.add(Dropout(0.95))
    model.add(Activation('relu'))
    
    # FC Layer 4:
    # 50 -> Dense -> 10
    model.add(Dense(10))
    model.add(Dropout(0.95))
    model.add(Activation('relu'))
    
    # Output Layer:
    # 10 -> Dense -> 1
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    model.fit_generator(train_generator, 
                        samples_per_epoch = len(train_samples), 
                        validation_data = validation_generator, 
                        nb_val_samples = len(validation_samples), 
                        nb_epoch = 10)

    model.save('model.h5')


def main(_):
    train()
    
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

