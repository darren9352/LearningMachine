"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

module_dir = os.path.dirname(__file__)
from cleverhans.attacks import FastGradientMethod
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', 'inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', 'dataset/images', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', 'output', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath,'rb') as f:
            image = np.array(Image.open(f).convert('RGB').resize((299, 299))).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):

    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            print('shape:', img.shape)
            Image.fromarray(img).save(f, format='PNG')



class InceptionModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
              x_input, num_classes=self.num_classes, is_training=False,
              reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        #print(probs)
        return probs

def fgsm_attack():
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [1, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        model = InceptionModel(num_classes)
        probs = model(x_input)

        for i in range(3):
            fgsm = FastGradientMethod(model)
            x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=os.path.join(module_dir,'inception_v3.ckpt'),
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(os.path.join(module_dir,FLAGS.input_dir), batch_shape):
                for i in range(20):
                    if i == 0:
                        adv_images = sess.run(x_adv, feed_dict={x_input: images})
                    else:
                        adv_images = sess.run(x_adv, feed_dict={x_input: adv_images})

                prob = sess.run(probs, feed_dict={x_input: images})

                original_idx = np.argmax(prob)
                print('original idx:', original_idx, 'prob:', prob[0][original_idx])
                adv_prob = sess.run(probs, feed_dict={x_input: adv_images})
                print(prob.shape)
                adv_idx = np.argmax(adv_prob)
                print('adv idx:', adv_idx, 'prob:', adv_prob[0][adv_idx])
                save_images(adv_images, filenames, os.path.join(module_dir,FLAGS.output_dir))




if __name__ == '__main__':
    tf.app.run()
