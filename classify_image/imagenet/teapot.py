
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
from cleverhans.attacks import FastGradientMethod

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
    img = image.array_to_img(img).copy()
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



def attack(img, d, eps):
    adversarial = img.copy()
    target = to_categorical([TARGET_CLASS], num_classes=1000)
    wrap = KerasModelWrapper(d)

    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': eps, 'clip_min': -1.0, 'clip_max': 1.0,
                     'y_target': target}

    adversarial = fgsm.generate_np(adversarial,**fgsm_params)
    return adversarial[0]


def iterative_attack(img, d, max_iterations=10, eps=0.01):
    adversarial = attack(img,d,eps)
    for i in range(max_iterations):
        print(i)
        adversarial = attack(np.expand_dims(adversarial,axis=0),d,eps)
        #show_predictions(d,np.expand_dims(adversarial,axis=0))
    return adversarial


def show_predictions(d, x, n=3):
    preds = d.predict(x)
    print(decode_predictions(preds, top=n)[0])


def plot_results(input_image, adversarial, noise):
    fig = plt.figure()
    fig.suptitle('Original')
    plt.imshow(input_image)

    fig = plt.figure()
    fig.suptitle('Adversarial')
    plt.imshow(deprocess(adversarial))

    # the "noise" * 100 #notnoise
    fig = plt.figure()
    fig.suptitle('Difference*100')
    plt.imshow(deprocess(100*(noise)))

    plt.show()

def save_images(images, filename, output_dir):
    img = Image.fromarray(images.astype(np.uint8))
    img.save("res.png")
    #with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
    #    Image.fromarray(images).save(f, format='PNG')


input_image = image.load_img(IMAGE_PATH, target_size=(IMAGE_SIZE, IMAGE_SIZE))
x = np.expand_dims(preprocess(input_image),axis=0)


# what was it classified as originally?
d = discriminator()
show_predictions(d,x)


# iterative fast gradient sign attack
# original image, discriminator, number of iterations, epsilon
adversarial = iterative_attack(x,d,3)


# show the results.
print("************************************************")
print("Results:")
#print('shape:', adversarial)
show_predictions(d,np.expand_dims(adversarial,axis=0))
#svimg = (((adversarial+ 1.0) * 0.5) * 255.0).astype(np.uint8)
#resimg = Image.fromarray(svimg)
#svimg = deprocess(adversarial)
#img = Image.fromarray(svimg)
#resimg.save("test.png")
plot_results(input_image, adversarial, adversarial-x[0])
#save_images(deprocess(adversarial), 'res.png', 'output')
