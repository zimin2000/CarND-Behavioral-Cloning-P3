import os, sys
import cv2
import numpy as np
from keras.models import load_model

model = load_model("model.h5")

def predict_all(imgs):

    images = []

    for i in imgs:
        # The current image from the center camera of the car
        image = cv2.imread(i)

        #print image
    
        image_array = np.asarray(image)

        images.append(image_array)

        #print image_array.shape

    steering_angles = model.predict(np.array(images), batch_size = len(images))

    print steering_angles

predict_all(sys.argv[1:])
#
#throttle = 0.2
#print steering_angle
