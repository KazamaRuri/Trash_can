import  tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('Trash_model')
# model = keras.models.load_model('Trash_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)