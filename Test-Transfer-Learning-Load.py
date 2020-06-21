import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from PIL import Image
from numpy import asarray
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import VGG16

# include top should be False to remove the softmax layer
pretrained_model = VGG16(include_top=True, weights='imagenet')

model2 = Sequential()
model2.add(Flatten(input_shape=(7,7,512)))
model2.add(Dense(100, activation='relu'))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
model2.add(Dense(10, activation='softmax'))

#model2.load_weights('imagenette2_model_weights.h5')
# model2.summary()
class_names = ["tench","springer","casette_player","chain_saw","church","French_horn","garbage_truck","gas_pump","golf_ball", "parachute"]
vgg_labels = np.loadtxt('synset.txt', str, delimiter='\t')

image = Image.open("F:/b.jpg")
image = image.resize((224, 224))
image = asarray(image)
image = tf.reshape(image, [-1, 224, 224, 3])
#test_image = image / 255
predictions = pretrained_model.predict(image)
#print(np.argmax(predictions))
print("Model prediction: %s" % vgg_labels[np.argmax(predictions)])
