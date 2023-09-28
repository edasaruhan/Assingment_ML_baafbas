import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

mnist = keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

train_images,test_images = train_images/255.0,test_images/255.0

train_labels = keras.utils.to_categorical(train_labels,10)
test_labels = keras.utils.to_categorical(test_labels,10)

class CustomDenseLayer(layers.Layer):
	def __init__(self,units,activation=None):
		super(CustomDenseLayer,self).__init__()
		self.units = units
		self.activation=activation

	def build(self,input_shape):
		self.kernel = self.add_weight("kernel",(input_shape[-1],self.units))
		self.bias = self.add_weight("bias",(self.units))

	def call(self,inputs):
		output = tf.matmul(inputs,self.kernel) + self.bias
		if self.activation is not None:
			output = self.activation(output)
		return output

model = keras.Sequential([
	layers.Flatten(input_shape(28,28)),
	CustomDenseLayer(128,activation=tf.nn.relu)
	CustomDenseLayer(64,activation=tf.nn.relu)
	CustomDenseLayer(10,activation=tf.nn.softmax)
	])
