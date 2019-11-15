import tensorflow as tf 
from tensorflow import keras 
import numpy as np 

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
def decode_review(data, text):
    return " ".join([get_reverse_word_index(data).get(i, "?") for i in text])
