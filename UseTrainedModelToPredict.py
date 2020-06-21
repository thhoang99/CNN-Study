import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
from numpy import asarray
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import VGG16

# include top should be False to remove the softmax layer
pretrained_model = VGG16(include_top=True, weights='imagenet')

vgg_labels = np.loadtxt('synset.txt', str, delimiter='\t')

image = Image.open("b.jpg")
image = image.resize((224, 224))
image = asarray(image)
image = tf.reshape(image, [-1, 224, 224, 3])

predictions = pretrained_model.predict(image)

print("Model prediction: %s" % vgg_labels[np.argmax(predictions)])
