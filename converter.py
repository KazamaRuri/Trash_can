import  tensorflow as tf
from tensorflow import keras

# model = keras.models.load_model('Saved_model')
model = keras.models.load_model('Saved_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model_quant)