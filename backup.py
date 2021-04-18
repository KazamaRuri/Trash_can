import  tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss')<0.4):
            print("\nLoss is low so cancelling training!")
            self.model.stop_training=True
callbacks = myCallback()

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
train_images_scaled=train_images/255
test_images_scaled=test_images/255

#-----------------------------------------------

model = keras.Sequential()
model.add(keras.layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512,activation=tf.nn.relu))
model.add(keras.layers.Dense(2,activation=tf.nn.softmax))
#model.summary()

#model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
#model.fit(train_images_scaled.reshape(-1,28,28,1),train_labels,epochs=1,callbacks=[callbacks])

model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.categorical_crossentropy,metrics=['accuracy'])
model.fit(train_generator,epochs=5)

#----------------------------------------

# model.evaluate(validation_generator)
print(np.argmax(model.predict(test_images_scaled[8].reshape(1,28,28,1))))
print(test_labels[8])

plt.imshow(train_generator[1,:,:,:])
plt.show()
