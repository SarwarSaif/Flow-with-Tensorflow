import tensorflow as tf 
from tensorflow import keras 
import numpy as np 

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

def get_word_index(data):
    word_index = data.get_word_index()

    word_index = {k:(v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] =2 
    word_index["<UNUSED>"] = 3
    
    return word_index

def get_reverse_word_index(data):
    word_index = get_word_index(data)
    
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return reverse_word_index

## Decode those integer values into texts
def decode_review(text):
    return " ".join([get_reverse_word_index(data).get(i, "?") for i in text])

## Creting Equal Shapes of Data 
word_index = get_word_index(data)
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen = 250)

## Check if shapes are equal 
print(len(test_data[0])==len(test_data[1]))

### Create Model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitmodel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

## Save the Model
model.save("test_classifier_model.h5")

"""
## Predict a data and show 
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ", decode_review(test_review))
print("Prediction: "+ str(predict[0]))
print("Actual: "+ str(test_labels[0]))
print(results)
"""