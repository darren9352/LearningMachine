import io
import os
import enum

#from base64 import b64decode
import base64
import tensorflow as tf
from PIL import Image
from django.core.files.temp import NamedTemporaryFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from classify_image.imagenet.fgsm import *
from classify_image.mnist.mnist_attack import *

from pathlib import Path
current_dir = os.path.dirname(__file__)
from tensorflow.contrib.slim.nets import inception

from keras.utils import to_categorical
#from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.layers.core import K
from keras import backend

import sys
tf.logging.set_verbosity(tf.logging.ERROR)

def clean_directory() :
	mnist_input_file = Path(os.path.join(current_dir, 'mnist/dataset/images/testtest.png'))
	mnist_output_file = Path(os.path.join(current_dir, 'mnist/output/testtest.png'))
	my_input_file = Path(os.path.join(current_dir, 'imagenet/dataset/images/testtest.png'))
	my_output_file = Path(os.path.join(current_dir, 'imagenet/output/testtest.png'))
	if my_input_file.is_file():
		os.remove(os.path.join(current_dir, 'imagenet/dataset/images/testtest.png'))
	if my_output_file.is_file():
		os.remove(os.path.join(current_dir, 'imagenet/output/testtest.png'))
	if mnist_input_file.is_file():
		os.remove(os.path.join(current_dir, 'mnist/dataset/images/testtest.png'))
	if mnist_output_file.is_file():
		os.remove(os.path.join(current_dir, 'mnist/output/testtest.png'))

@csrf_exempt
def classify_api(request):
	data = {"success": False}
	clean_directory()

	if request.method == "POST":
		model = request.POST.get("model", None)
		if model == 'imagenet':
			tmp_f = NamedTemporaryFile()
			tmp_adver = NamedTemporaryFile()

			if request.FILES.get("image", None) is not None:
				image_request = request.FILES["image"]
				image_bytes = image_request.read()
				image.save(tmp_f, image.format)

			elif request.POST.get("image64", None) is not None:
				base64_data = request.POST.get("image64", None).split(',', 1)[1]
				plain_data = base64.b64decode(base64_data)
				image = Image.open(io.BytesIO(plain_data))
				image.save(os.path.join(current_dir, 'imagenet/dataset/images/testtest.png'))
				tmp_f.write(plain_data)

			tmp_f.close()

			# Backend session for attack
			print('Building Backend Session.')
			K.set_learning_phase(0)
			sess = tf.Session()
			backend.set_session(sess)

			# Image preprocess
			print('Modifying image')
			x = np.expand_dims(preprocess(image.resize((299, 299))),axis=0)
			img_shape = [1, 299, 299, 3]
			x_input = tf.placeholder(tf.float32, shape=img_shape)

			# Define model
			d = discriminator()

			# Prediction of original image
			print('prediction of original image')
			classify_result = get_predictions(d, x, 10)

			# Select attack algorithm and iteration

			attack_algorithm = request.POST.get("attack", None)
			n = int(request.POST.get("iterate", None))

			# Start attack
			result, attack_speed = attack(attack_algorithm, n, d, x_input, x, sess)
			print("--- %s seconds ---" %(attack_speed))
			print('classified by ', result[0][1])

			# Print image to web site
			with open(os.path.join(current_dir,'imagenet/output/testtest.png'), 'rb') as img_file:
				img_str = base64.b64encode(img_file.read())
			tmp_adver.write(base64.b64decode(img_str))
			tmp_adver.close()
		elif model == 'mnist':
			tmp_adver = NamedTemporaryFile()
			tmp_f = NamedTemporaryFile()
			mnist_sample = int(request.POST.get("sample", None))
			mnist_target = int(request.POST.get("target", None))
			mnist_algorithm = request.POST.get("mnist_algorithm", None)
			result, attack_speed = mnist_attack_func(mnist_sample, mnist_target, mnist_algorithm)
			print("--- %s seconds ---" %(attack_speed))
			print('classified by', np.argmax(result))

			result = result.tolist()
			with open(os.path.join(current_dir,'mnist/dataset/images/testtest.png'), 'rb') as input_file:
				input_str = base64.b64encode(input_file.read())
			tmp_f.write(base64.b64decode(input_str))
			tmp_f.close()
			with open(os.path.join(current_dir,'mnist/output/testtest.png'), 'rb') as img_file:
				img_str = base64.b64encode(img_file.read())
			tmp_adver.write(base64.b64decode(img_str))
			tmp_adver.close()

		# Make Graph
		data["attack_speed"] = attack_speed
		data["success"] = True
		data["confidence"] = {}
		if model == 'imagenet':
			data["model"] = 'imagenet'
			for i in range(len(classify_result)) :
				data["confidence"][classify_result[i][1]] = float(classify_result[i][2])
			data["adverimage"] = 'data:image/png;base64,' + img_str.decode('utf-8')
			data["adversarial"] = {}
			for i in range(len(result)) :
				data["adversarial"][result[i][1]] = float(result[i][2])
				#print('iter:', i, 'name:', result[i][1], 'pred:', result[i][2])

			sess.close()

		elif model == 'mnist':
			data["model"] = 'mnist'
			for i in range(10):
				if i == mnist_sample:
					data["confidence"][str(i)] = float(1)
				else:
					data["confidence"][str(i)] = float(0)
			data["input_image"] = 'data:image/png;base64,' + input_str.decode('utf-8')
			data["adverimage"] = 'data:image/png;base64,' + img_str.decode('utf-8')
			data["adversarial"] = {}
			for i in range(len(result[0])) :
				data["adversarial"][str(i)] = float(result[0][i])


		# Close the session
		# sess.close()
	return JsonResponse(data)

def classify(request):
	return render(request, 'classify.html', {})
