import  tensorflow as tf
from tensorflow import keras

# model = keras.models.load_model('Trash_model')
model = keras.models.load_model('Trash_model.h5')

conv = tf.lite.TFLiteConverter.from_keras_model(model)
# converter = conv.convert()

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8  # or tf.uint8
# # converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = conv.convert()

# Save the model.e
with open('model_optimization.tflite', 'wb') as f:
  f.write(tflite_model)