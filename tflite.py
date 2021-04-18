import numpy as np
import math
import time
from tflite_runtime.interpreter import Interpreter
from PIL import Image


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

interpreter = Interpreter("./model_optimization.tflite")



interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']


# print(height,width)

image = Image.open("./4.jpg")
img = np.array(image.resize((128, 128), resample=Image.LANCZOS))

results = classify_image(interpreter, img)


print(results,type(results))
