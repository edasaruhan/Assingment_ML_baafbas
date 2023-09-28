import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

mnist = keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()