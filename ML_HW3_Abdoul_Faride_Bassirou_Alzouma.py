import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

mnist = keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

train_images,test_images = train_images/255.0,test_images/255.0

train_labels = keras.utils.to_categorical(train_labels,10)
test_labels = keras.utils.to_categorical(test_labels,10)

