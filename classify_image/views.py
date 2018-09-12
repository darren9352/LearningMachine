import io
import os

#from base64 import b64decode
import base64
import tensorflow as tf
from PIL import Image
from django.core.files.temp import NamedTemporaryFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from classify_image.imagenet.fgsm import *
from pathlib import Path
current_dir = os.path.dirname(__file__)
from tensorflow.contrib.slim.nets import inception

MAX_K = 10

TF_GRAPH = "{base_path}/inception_model/graph.pb".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
TF_LABELS = "{base_path}/inception_model/labels.txt".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))


def load_graph():
    sess = tf.Session()
    with tf.gfile.FastGFile(TF_GRAPH, 'rb') as tf_graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(tf_graph.read())
        tf.import_graph_def(graph_def, name='')
    label_lines = [line.rstrip() for line in tf.gfile.GFile(TF_LABELS)]
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    return sess, softmax_tensor, label_lines


SESS, GRAPH_TENSOR, LABELS = load_graph()


@csrf_exempt
def classify_api(request):
    data = {"success": False}
    my_input_file = Path(os.path.join(current_dir, 'imagenet/dataset/images/testtest.png'))
    my_output_file = Path(os.path.join(current_dir, 'imagenet/output/testtest.png'))
    if my_input_file.is_file():
        os.remove(os.path.join(current_dir, 'imagenet/dataset/images/testtest.png'))
    if my_output_file.is_file():
        os.remove(os.path.join(current_dir, 'imagenet/output/testtest.png'))
    if request.method == "POST":
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

        classify_result = tf_classify(tmp_f, int(request.POST.get('k', MAX_K)))
        tmp_f.close()
        fgsm_attack()
        with open(os.path.join(current_dir,'imagenet/output/testtest.png'), 'rb') as img_file:
            img_str = base64.b64encode(img_file.read())
        tmp_adver.write(base64.b64decode(img_str))
        adver_result = tf_classify(tmp_adver, int(request.POST.get('k', MAX_K)))
        tmp_adver.close()

        if classify_result:
            data["success"] = True
            data["confidence"] = {}
            for res in classify_result:
                data["confidence"][res[0]] = float(res[1])

        if adver_result:
            data["adverimage"] = 'data:image/png;base64,' + img_str.decode('utf-8')
            data["adversarial"] = {}
            for res in adver_result:
                data["adversarial"][res[0]] = float(res[1])

    return JsonResponse(data)


def classify(request):
    return render(request, 'classify.html', {})


# noinspection PyUnresolvedReferences
def tf_classify(image_file, k=MAX_K):
    result = list()
    image_data = tf.gfile.FastGFile(image_file.name, 'rb').read()
    predictions = SESS.run(GRAPH_TENSOR, {'DecodeJpeg/contents:0': image_data})
    predictions = predictions[0][:len(LABELS)]
    top_k = predictions.argsort()[-k:][::-1]
    for node_id in top_k:
        label_string = LABELS[node_id]
        score = predictions[node_id]
        result.append([label_string, score])

    return result
