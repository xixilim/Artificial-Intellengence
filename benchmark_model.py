"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Loading the CIFAR-10 datasets
from tensorflow.keras.datasets import cifar10

import ssl
# ssl certification required
ssl._create_default_https_context = ssl._create_unverified_context

# set the start time 
start_time = datetime.datetime.now()

# gpus information setting up
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# batch_size equals to 256 leads to a more efficient running time
batch_size = 256

# classes number of pictures are labeled to 0-9
num_classes = 10
# relativly small epochs number
epochs = 100

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# x_train - training data(images), y_train - labels(digits)
# Print figure with 10 random images from each

# normalize dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# label every pictures different name 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# preview pixels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Convert and pre-processing

x_train = x_train.reshape((50000, 32, 32, 3))
x_train = x_train.astype('float32') / 255.

x_test = x_test.reshape((10000, 32, 32, 3))
x_test = x_test.astype('float32') / 255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# define model with 3 convolutional layers and 1 fully connect layer
def base_model():
    model = Sequential()
    # 3 convolutional layers
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (4, 4), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(32, (4, 4), activation='relu', kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    # 1 fully connect layer
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model


# model output & summary
cnn_n = base_model()
cnn_n.summary()

# Fit model


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# training model with dataset and defined parameters
cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose = 0)

# evaluate model
_, acc = cnn_n.evaluate(x_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Plots for training and testing process: loss and accuracy

# accuracy plot
plt.figure(0)
plt.plot(cnn.history['accuracy'], 'r')
plt.plot(cnn.history['val_accuracy'], 'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])

# loss plot
plt.figure(1)
plt.plot(cnn.history['loss'], 'r')
plt.plot(cnn.history['val_loss'], 'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])

plt.show()

from sklearn.metrics import classification_report, confusion_matrix

# prediction part  
Y_pred = cnn_n.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

# confusion matrix content creation
for ix in range(10):
    print(ix, confusion_matrix(np.argmax(y_test, axis=1), y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(cm)

# Visualizing of confusion matrix
import seaborn as sn
import pandas as pd

df_cm = pd.DataFrame(cm, range(10),
                     range(10))
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()

# calculate running time
stop_time = datetime.datetime.now()
print("Time required for training model ", ': ', stop_time - start_time)
