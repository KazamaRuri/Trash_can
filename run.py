import  tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
from PIL import Image

model = keras.models.load_model('Trash_model')
# model = keras.models.load_model('Trash_model.h5')

img = np.array(Image.open('test2.jpg'))
# plt.imshow(img)
# plt.show()

sort = np.argmax(model.predict(img.reshape(1,128,128,3)))

print(model.predict(img.reshape(1,128,128,3)))

print(sort)
if sort==0:
    print("厨余垃圾")
elif sort==1:
    print("废旧电池")
elif sort==2:
    print("矿泉水瓶")
elif sort==3:
    print("烟头")
elif sort==4:
    print("易拉罐")