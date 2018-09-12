
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import other_classes, set_log_level
from cleverhans_tutorials.tutorial_models import ModelBasicCNN

from mnist_handle import get_mnist_data
from mnist_handle import get_mnist_idx

FLAGS = flags.FLAGS

def get_mnist_idx(y_test, want_class) :
    for i in range(10000) :
        if np.argmax(y_test[i]) == want_class :
            return i

def simulate_jsma():
    # MNIST-specific dimensions
    # img_rows = 28
    # img_cols = 28
    # channels = 1

    # Get MNIST test data
    x_test, y_test = get_mnist_data()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = ModelBasicCNN('model1', 10, 64)
    preds = model.get_logits(x)
    print("Defined TensorFlow model graph.")

    ##################################
    #          Load Model            #
    ##################################
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        save_path = r'C:\Users\hhkim\Desktop\scc\cleverhans\cleverhans_tutorials\model_data\train_model'
        saver.restore(sess, save_path)

        sample_class = int(input('input sample class(0-9): '))
        target_class = int(input('input target class(0-9): '))

        if sample_class<0 or sample_class>9 or target_class<0 or target_class>9 :
            print('input is wrong')

        sample_idx = get_mnist_idx(y_test, sample_class)
        target_idx = get_mnist_idx(y_test, target_class)

        sample = x_test[sample_idx:sample_idx+1]

        # Instantiate a SaliencyMapMethod attack object
        jsma = SaliencyMapMethod(model, back='tf', sess=sess)
        jsma_params = {'theta': 1., 'gamma': 0.1,
                    'clip_min': 0., 'clip_max': 1.,
                    'y_target': None}

        jsma_params['y_target'] = y_test[target_idx:target_idx+1]
        adv_x = jsma.generate_np(sample, **jsma_params)

        print('sample class:', np.argmax(y_test[sample_idx]))
        print('target class:', np.argmax(y_test[target_idx]))

        # Get array of output
        # updated by hak hyun kim
        feed_dict = {x: adv_x}
        probabilities = sess.run(preds, feed_dict)

        print('==========================================')
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        print(softmax(probabilities))
        #print(probabilities)
        print('==========================================')

        print('{} class is recognized by {} '.format(sample_class, target_class))


    # save the adverisal image #
    two_d_img = (np.reshape(adv_x, (28, 28)) * 255).astype(np.uint8)
    from PIL import Image
    save_image = Image.fromarray(two_d_img)
    save_image = save_image.convert('RGB')
    save_image.save("res.png")

    sess.close()
    #return report

def main(argv=None):
    simulate_jsma()

if __name__ == '__main__':
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')

    tf.app.run()
