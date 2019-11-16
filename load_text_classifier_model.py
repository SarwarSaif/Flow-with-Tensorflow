import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import decode_words as tc 

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

word_index = tc.get_word_index(data)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen = 250)

def encode_review(str):
    encoded = [1]
    
    for word in str:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
            
    return encoded
model = keras.models.load_model("test_classifier_model.h5")

with open("test_data_for_text_classifier.txt", encoding="utf-8") as file:
    for line in file.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(","").replace(")","").replace("\"","").strip().split(" ")
        encode = encode_review(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen = 250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
"""        
results = model.evaluate(test_data, test_labels)

## Predict a data and show 
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ",tc.decode_review(data, test_review))
print("Prediction: "+ str(predict[0]))
print("Actual: "+ str(test_labels[0]))
print(results)

"""


