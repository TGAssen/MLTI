import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn.preprocessing as skpre


#load data
cifar10= tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


#grayscale images
train_images=np.array([cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)for image in train_images])
test_images=np.array([cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)for image in test_images])


#normalize images
train_images = train_images/255.0
test_images = test_images / 255.0

#label preprocessing
one_hot_encoder = skpre.OneHotEncoder(sparse=False)
one_hot_encoder.fit(train_labels)
train_labels= one_hot_encoder.transform(train_labels)
test_labels= one_hot_encoder.transform(test_labels)

#model based on textbook 3.3.1
model= tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same', input_shape=(32,32,1)))
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,activation='relu'))
#helps against overfitting
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

#compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#define early stopping object
es= tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', mode='min', verbose=1, patience=3)
#train model
history = model.fit(train_images,train_labels, epochs= 10, batch_size=32, validation_data=(test_images,test_labels),callbacks=[es])

#Evaluate model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label= 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5,1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose='2')
print(test_acc)

