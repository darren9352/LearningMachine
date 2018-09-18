
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.layers.core import K
from keras import backend

import tensorflow as tf

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import SaliencyMapMethod

IMAGE_SIZE=299
TARGET_CLASS=849 # teapot
#TARGET_CLASS=1 # goldfish
IMAGE_PATH="img/01f824264783f58d.png"

K.set_learning_phase(0)
sess = tf.Session()
backend.set_session(sess)

def deprocess(input_image):
    img = input_image.copy()
    img /= 2.
    img += 0.5
    img *= 255. # [-1,1] -> [0,255]
    #img = image.array_to_img(img).copy() 
    return img

def preprocess(input_image):
    img = image.img_to_array(input_image).copy()
    img /= 255.
    img -= 0.5
    img *= 2. # [0,255] -> [-1,1]
    return img

def discriminator():
    model = InceptionV3(weights='imagenet')
    return model

def show_predictions(d, x, n=3):
    preds = d.predict(x)
    print(decode_predictions(preds, top=n)[0])

def attack(model, x_input, input_img):
    wrap = KerasModelWrapper(model)
    jsma = SaliencyMapMethod(wrap, sess=sess)
    jsma_params = {'theta': 0.1, 'gamma': 0.1,
                    'clip_min': -1., 'clip_max': 1.,
                    'y_target': None}
    adv_x = jsma.generate_np(input_img, **jsma_params)
    return adv_x

input_image = image.load_img(IMAGE_PATH, target_size=(IMAGE_SIZE, IMAGE_SIZE)) 
x = np.expand_dims(preprocess(input_image),axis=0)

img_shape = [1, IMAGE_SIZE, IMAGE_SIZE, 3]
x_input = tf.placeholder(tf.float32, shape=img_shape)

# what was it classified as originally?
d = discriminator()
show_predictions(d,x)


import time
start_time = time.time() 
print('attack is start.')
res = attack(d, x_input, x)
print('attack is end.')
print("--- %s seconds ---" %(time.time() - start_time))

# show the results.
print("************************************************")
print("Results:")
#show_predictions(d,np.expand_dims(adversarial,axis=0))

preds = d.predict(res)
print(decode_predictions(preds, top=3)[0])

d_img = deprocess(res[0]).astype(np.uint8)
sv_img = Image.fromarray(d_img)
sv_img.save("./output/jsma_res.png")


