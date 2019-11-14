import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## Regularization of Data
train_images = train_images/255.0
test_images = test_images/255.0

## Show Dataas Image in Matplotlib
##plt.imshow(train_images[7], cmap = plt.cm.binary)
##plt.show()

## Create Model with keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="sigmoid"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=15)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Accuracy: ", test_acc)


