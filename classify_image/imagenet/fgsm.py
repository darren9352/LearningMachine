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
from cleverhans.attacks import *

IMAGE_SIZE=299
TARGET_CLASS=849 # teapot
#TARGET_CLASS=1 # goldfish
IMAGE_PATH="img/01f824264783f58d.png"

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

def show_predictions(d, x, n=10):
    preds = d.predict(x)
    print(decode_predictions(preds, top=n)[0])

def get_predictions(d, x, n=10) :
    preds = d.predict(x)
    return decode_predictions(preds, top=10)[0]

def cw_attack_keras(model, x_input, input_img, sess):
    wrap = KerasModelWrapper(model)
    
    cw_params = {'binary_search_steps': 1,
                    'max_iterations': 5,
                    'learning_rate': 2e-3,
                    'batch_size': 1,
                    'initial_const': 0.1,
                    'confidence' : 0,
                    'clip_min': -1.,
                    'clip_max': 1.}

    cw = CarliniWagnerL2(wrap, sess=sess)
    adv = cw.generate(x=x_input, initial_const=2.0 * 16 / 255.0, batch_size = 1,
		 binary_search_steps= 4, learning_rate=2e-3, clip_min=-1., clip_max=1.,
		 max_iterations=5)
    adv_img = sess.run(adv, feed_dict={x_input: input_img})
    return adv_img


def attack(model, x_input, input_img, sess):
    wrap = KerasModelWrapper(model)
    eps = 2.0 * 16 / 255.0
    #eps = 0.1 
    fgsm = FastGradientMethod(wrap)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)
    adv_image = sess.run(x_adv, feed_dict={x_input: input_img})

    for i in range(3):
        if i == 0:
            adv_image = sess.run(x_adv, feed_dict={x_input: input_img})
        else:
            adv_image = sess.run(x_adv, feed_dict={x_input: adv_image})
    return adv_image

	
def fgsm_attack(d, x_input, x, sess) :
    """
    K.set_learning_phase(0)
    sess = tf.Session()
    backend.set_session(sess)
    print('make backend session.')
    
    print('loading target image...')
    current_dir = os.path.dirname(__file__)
    #input_image = image.load_img("img/01f824264783f58d.png") 
    path = os.path.join(current_dir, 'dataset/images/testtest.png')
    input_image = Image.open(path) 
    x = np.expand_dims(preprocess(input_image),axis=0)
    img_shape = [1, 299, 299, 3]
    x_input = tf.placeholder(tf.float32, shape=img_shape)
    
    d = discriminator()

    show_predictions(d,np.expand_dims(preprocess(input_image), axis=0))
    """
    current_dir = os.path.dirname(__file__)

    res = attack(d, x_input, x, sess)

	# show the results.
    print("************************************************")
    print("Results:")
    #show_predictions(d,np.expand_dims(res,axis=0))
    
    preds = d.predict(res)
    #print(decode_predictions(preds, top=10)[0])

    d_img = deprocess(res[0]).astype(np.uint8)
    sv_img = Image.fromarray(d_img)
    path = os.path.join(current_dir, 'output/testtest.png')
    sv_img.save(path)

    return decode_predictions(preds, top=10)[0]

def cw_attack(d, x_input, x, sess) :

    res = cw_attack_keras(d, x_input, x, sess)

    preds = d.predict(res)
    #print(decode_predictions(preds, top=3)[0])

    d_img = deprocess(res[0]).astype(np.uint8)
    sv_img = Image.fromarray(d_img)
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, 'output/testtest.png')
    sv_img.save(path)

    return decode_predictions(preds, top=10)[0]
