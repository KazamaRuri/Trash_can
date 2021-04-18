import  tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
from PIL import Image
import time

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(128,128),
    batch_size=32
)


# validation_datagen = ImageDataGenerator(rescale=1/255)
# validation_generator = validation_datagen.flow_from_directory(
#     'validation',
#     target_size=(128,128),
#     batch_size=32,
#     # class_mode='sparse'
# )

model = keras.Sequential()
model.add(keras.layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,activation=tf.nn.relu))
model.add(keras.layers.Dense(5,activation=tf.nn.softmax))
#model.summary()

start = time.perf_counter()

model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.categorical_crossentropy,metrics=['accuracy'])
model.fit(train_generator,epochs=10)

# model.save('Trash_model')
# model.save('Trash_model.h5')


# model.evaluate(validation_generator)

img = np.array(Image.open('test2.jpg'))
# plt.imshow(img)
# plt.show()

# sort = np.argmax(model.predict(img.reshape(1,128,128,3)))

print(model.predict(img.reshape(1,128,128,3),batch_size=32,verbose=1))

stop = time.perf_counter() - start

print(stop)

# print(sort)
# if sort==1:
#     print("废旧电池")
# elif sort==2:
#     print("矿泉水瓶")
# elif sort==3:
#     print("烟头")
# elif sort==4:
#     print("易拉罐")
# else:
#     print("厨余垃圾")