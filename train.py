import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

#---Model---#

model = keras.Sequential()
model.add(keras.layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=(128,128,3)))  #input=*128*128*3
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,activation=tf.nn.relu))
model.add(keras.layers.Dense(5,activation=tf.nn.softmax))
#model.summary()

#---Model---#

#---ImageDataGenerator---#

train_datagen = ImageDataGenerator()    #train_image
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(128,128),
    batch_size=32
)

# validation_datagen = ImageDataGenerator()    #validation_img
# validation_generator = validation_datagen.flow_from_directory(
#     'validation',
#     target_size=(128,128),
#     batch_size=32
# )

#---ImageDataGenerator---#

model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.categorical_crossentropy,metrics=['accuracy'])
model.fit(train_generator,epochs=1)

# model.save('Saved_model')
# model.save('Saved_model.h5')

# model.evaluate(validation_generator)

image = Image.open("test_img/test.jpg")
img = np.array(image.resize((128, 128), resample=Image.LANCZOS))

results = model.predict(img.reshape(1,128,128,3))
sort = np.argmax(results)
print(results,type(results))