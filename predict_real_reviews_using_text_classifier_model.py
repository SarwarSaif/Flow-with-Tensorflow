import tensorflow as tf 
from tensorflow import keras 

data = keras.datasets.imdb
word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] =2 
word_index["<UNUSED>"] = 3

model = keras.models.load_model("test_classifier_model.h5")
def encode_review(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
            
    return encoded

with open("test_data_for_text_classifier.txt", encoding="utf-8") as file:
    for line in file.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = encode_review(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
        print("Length of data ", len(predict))