# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:29:24 2022

@author: chenl
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import sys
import matplotlib.pyplot as plt
import re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from pickle import dump
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras as K
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Flatten, Dropout
import datetime

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

embed_size = 128

max_features = 2000
vocab_size = 3000

lstm_out = 64

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data = pd.read_csv("train.csv")
data.drop("id", axis=1, inplace=True)
print("Rows:",data.shape[0]," Columns:",data.shape[1])

# amount of racist vs non-racist tweets
print(data["label"].value_counts())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = text.strip(' ')
    return text

data['tweet'] = data['tweet'].map(lambda t: clean_text(t))
data['tweet'] = data['tweet'].map(lambda t: clean_text(t))


tokenizer = Tokenizer(num_words = max_features, split=' ')
tokenizer.fit_on_texts(data['tweet'].values)
x = tokenizer.texts_to_sequences(data['tweet'].values)
x = pad_sequences(x)

y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 4)

print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()

# define model
model = Sequential()

#model.add(Embedding(vocab_size*2, embed_size, input_shape=(x_train.shape[1],)))
#model.add(Embedding(1500, embed_size, input_shape=(x_train.shape[1],)))
model.add(Embedding(vocab_size, embed_size, input_shape=(x_train.shape[1],)))
model.add(K.layers.LSTM(units=1024, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False, return_sequences=True))
model.add(K.layers.LSTM(units=512, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False))
#model.add(K.layers.LSTM(units=256, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False))
model.add(K.layers.Dense(units=10000, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='sigmoid'))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the model
history = model.fit(x_train, y_train, epochs=80, batch_size = 256, validation_split = 0.2, verbose=0)
# evaluate the model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# save the model to file
model.save('model_twitter.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer_m2.pkl', 'wb'))
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)

# loss plot
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('PReLU training / validation loss values')
plt.ylabel('Loss value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()


# accuracy plot
plt.figure(0)
plt.plot(history.history['accuracy'], 'r')
plt.plot(history.history['val_accuracy'], 'g')
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])



# read model twitter
import h5py
filename = "model_twitter.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    # print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    print(data)
