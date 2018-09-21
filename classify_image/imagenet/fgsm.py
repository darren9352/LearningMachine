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

def save_image(adv_img, noise, path):
    # Save adversarial image
    d_img = deprocess(adv_img[0]).astype(np.uint8)
    sv_img = Image.fromarray(d_img)
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, path)
    sv_img.save(path)

    # Save noise
    d_img = deprocess(100*noise).astype(np.uint8)
    sv_img = Image.fromarray(d_img)
    path = os.path.join(current_dir, 'output/noise.png')
    sv_img.save(path)

def deepfool_attack(model, n, x_input, input_img, sess):
    import time
    start_time = time.time() 
    wrap = KerasModelWrapper(model)
    deepfool = DeepFool(wrap, sess=sess)
    adv_x = deepfool.generate(x=x_input, over_shoot=0.02, max_iter=n,
        nb_candidate=2, clip_min=-1., clip_max=1.)
    adv_img = sess.run(adv_x, feed_dict={x_input: input_img})
    attack_time = time.time() - start_time

    noise = input_img[0] - adv_img[0]
    #save_image(noise)
    save_image(adv_img, noise, 'output/testtest.png')
    preds = model.predict(adv_img)
    return decode_predictions(preds, top=10)[0], attack_time

def cw_attack_keras(model, x_input, input_img, sess, n):
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
		 max_iterations=n)
    adv_img = sess.run(adv, feed_dict={x_input: input_img})
    return adv_img


def fgsm_attack_iter(model, x_input, input_img, sess, n):
    wrap = KerasModelWrapper(model)
    eps = 2.0 * 16 / 255.0
    fgsm = FastGradientMethod(wrap)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)
    adv_image = sess.run(x_adv, feed_dict={x_input: input_img})

    for i in range(n):
        if i == 0:
            adv_image = sess.run(x_adv, feed_dict={x_input: input_img})
        else:
            adv_image = sess.run(x_adv, feed_dict={x_input: adv_image})
    return adv_image


def attack(algorithm, n, d, x_input, x, sess):
    print(algorithm, 'attack is start')
    if algorithm == 'FGSM':
        result, attack_time = fgsm_attack(d, n, x_input, x, sess)
    elif algorithm == 'CWL2':
        result, attack_time = cw_attack(d, n, x_input, x, sess)
    elif algorithm == 'DeepFool' :
        result, attack_time = deepfool_attack(d, n, x_input, x, sess)
    return result, attack_time

def fgsm_attack(d, n, x_input, x, sess) :
    import time
    start_time = time.time() 
    res = fgsm_attack_iter(d, x_input, x, sess, n)
    attack_time = time.time() - start_time

    noise = res[0] - x[0]
    save_image(res, noise, 'output/testtest.png')
    preds = d.predict(res)
    return decode_predictions(preds, top=10)[0], attack_time

def cw_attack(d, n, x_input, x, sess) :
    import time
    start_time = time.time() 
    res = cw_attack_keras(d, x_input, x, sess, n)
    attack_time = time.time() - start_time

    noise = res[0] - x[0]
    save_image(res, noise, 'output/testtest.png')
    preds = d.predict(res)
    return decode_predictions(preds, top=10)[0], attack_time
