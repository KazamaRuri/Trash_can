import  tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
from PIL import Image

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    '',
    target_size=(224,224),
    batch_size=32,
    #class_mode='sparse'
)
#validation_generator = validation_datagen.flow_from_directory(
#    '',
#    target_size=(224,224),
#    batch_size=32,
#    #class_mode='sparse'
#)

model = keras.Sequential()
model.add(keras.layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=(224,224,3)))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(4,activation=tf.nn.softmax))
#model.summary()

model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.categorical_crossentropy,metrics=['accuracy'])
model.fit(train_generator,epochs=5)

#model.save('')

#model.evaluate(validation_generator)

img = np.array(Image.open(''))
#plt.imshow(img)
#plt.show()
print(np.argmax(model.predict(img.reshape(1,224,224,3))))