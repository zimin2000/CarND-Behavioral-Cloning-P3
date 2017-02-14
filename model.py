import cv2
import os
import csv
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout, Activation, SpatialDropout2D
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

BATCH_SIZE = 32

IMG_WIDTH = 320
IMG_HEIGHT = 160

CROP_TOP = 50
CROP_BOTTOM = 20
CROP_SIDE = 1

samples = []

def filter_out_tiny(data):
    """
    Remove about 70% of steering values below 0.03
    """

    return filter((lambda item: abs(item['steering']) < 0.03 and np.random.randint(100) <= 70), data)

def read_data(DATA_DIR, has_header=True):
    """
    Load CSV file into a list items. Each item is a dictionary or the name of 
    column mapped to value. List of columns is expected to be the first line 
    of the file.
    """
    data = []
    columns = ["center","left","right","steering","throttle","brake","speed"]

    with open(os.path.join(DATA_DIR, "driving_log.csv")) as FILE:
        reader = csv.reader(FILE)

        if has_header:
            if reader.next() != columns:
                raise Exception('Unexpected set of columns.')

        for values in reader:
            if len(values) != len(columns):
                raise Exception('Column number missmatch.')

            for i in range(3):
                values[i] = os.path.join(DATA_DIR, 'IMG', values[i].split('/')[-1])

            for i in range(3, 7):
                values[i] = float(values[i])

            data.append(dict(zip(columns, values)))

    return data

orig_samples = read_data("./data")
extra_samples = read_data("./extra_data", has_header=False)

samples = orig_samples + extra_samples

print len(samples)

samples = filter_out_tiny(samples)

print len(samples)

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
                choice = np.random.randint(100)

                if choice < 50:
                    image_name = sample['center']
                    angle = sample['steering']

                elif choice < 75:
                    image_name = sample['left']
                    angle = sample['steering'] + 0.20

                else:
                    image_name = sample['right']
                    angle = sample['steering'] - 0.20

                image = cv2.imread(image_name)

                # print "{} {} {}".format(name, center_image.shape, center_angle)

                choice = np.random.randint(100)

                if choice < 50:
                    image = image[:,::-1,:]
                    angle = -angle

                # 68% - dx=0, 16% dx = +1 or -1
                dx = np.clip(int(np.random.normal()),-1,1)
                dy = np.clip(int(np.random.normal()),-1,1)

                image = np.roll(np.roll(image, dx, axis=1), dy, axis=0)

                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield (X_train, y_train)



def build_model(input_shape):
    model = Sequential()

    # Preprocessing 0.1: Cropping useless top and bottom pixels
    # 3 @ 320 x 160 -> 3 @ 318 x 90
    model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (CROP_SIDE, CROP_SIDE)), input_shape=input_shape, dim_ordering='tf'))

    # Preprocessing 0.2: Centered around zero with small standard deviation
    # 3 @ 318 x 90 -> 3 @ 318 x 90
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))


    # Convolutional Layer 1:
    # 3 @ 318 x 90 -> Conv 7 x 7 (+1 x +1) -> MaxPool 3 x 3 -> Dropout -> Elu -> 24 @ 104 x 28
    model.add(Convolution2D(24, 7, 7, border_mode='valid'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Dropout(0.4))
    model.add(Activation('elu'))

    # Convolutional Layer 2:
    # 24 @ 104 x 28 -> Conv 5 x 5 (+1 x +1) -> MaxPool 2 x 2 -> Dropout -> Elu -> 36 @ 50 x 12
    model.add(Convolution2D(36, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('elu'))

    # Convolutional Layer 3:
    # 36 @ 50 x 12 -> Conv 5 x 5 (+1 x +1) -> MaxPool 2 x 2 -> Dropout -> Elu -> 48 @ 23 x 4
    model.add(Convolution2D(48, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.6))
    model.add(Activation('elu'))

    # Convolutional Layer 4:
    # 48 @ 23 x 4 -> Conv 3 x 3 -> Dropout -> Elu -> 64 @ 21 x 2
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Dropout(0.6))
    model.add(Activation('elu'))

    # Convolutional Layer 5:
    # 64 @ 21 x 2 -> Conv 3 x 2 -> Dropout -> Elu -> 64 @ 19 x 1
    model.add(Convolution2D(64, 2, 3, border_mode='valid'))
    model.add(Dropout(0.6))
    model.add(Activation('elu'))

    # FC Layer 1:
    # 64 @ 19 x 1 -> Flatten -> Dropout -> Elu -> ????
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Activation('elu'))
    
    # FC Layer 2:
    # 1164 -> Dropout -> Elu -> 100
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Activation('elu'))
    
    # FC Layer 3:
    # 100 -> Dropout -> Elu -> 50
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Activation('elu'))
    
    # FC Layer 4:
    # 50 -> Dropout -> 10
    model.add(Dense(10))
    model.add(Dropout(0.4))
    model.add(Activation('elu'))

    # Output Layer:
    # 10 -> Dense -> 1
    model.add(Dense(1))

    return model
    

def train():

    model = build_model((IMG_HEIGHT, IMG_WIDTH, 3))

    model.summary()

    model.compile(loss='mse', optimizer='adam')

    checkpointer = ModelCheckpoint(filepath="model.checkpoint.h5", verbose=1, save_best_only=True)

    model.fit_generator(generator = generator(train_samples, batch_size = BATCH_SIZE), 
                        samples_per_epoch = len(train_samples) * 10, 
                        validation_data = generator(validation_samples, batch_size = BATCH_SIZE), 
                        nb_val_samples = len(validation_samples) * 10, 
                        nb_epoch = 40,
                        callbacks = [checkpointer])

    model.save('model.h5')


def main(_):
    train()
    
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

