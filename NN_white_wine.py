"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import datetime
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
wine = pd.read_csv("winequality-white.csv", delimiter = ";", header = 0)
wine = wine.replace(np.nan,0)
wine = wine.values
print(wine.shape)

x = wine[:, 0:11]
y = wine[:, 11]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
np.random.seed(4)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, \
                                                    random_state = 4)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()

def origin_model():
    # create model
    model = Sequential()

    # The input layer
    model.add(Dense(35, input_dim = 11, activation = 'relu', \
                kernel_initializer ='normal'))

    # The Hidden Layer
    model.add(Dense(15, activation = 'relu', \
                kernel_initializer ='normal'))
    
    # The Output Layer
    model.add(Dense(1, kernel_initializer ='normal', activation = 'linear'))

    # compile the network
    model.compile(loss ='mean_squared_error', optimizer = 'adam', \
              metrics = ['mse', 'mae'])
    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section | K-fold validation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start = datetime.datetime.now()
k = 5
num_val_samples = len(y_train) // k
num_epochs = 50
bat_size = 50
all_train_loss = []
all_val_loss = []
for i in range(k):
    print('processing fold #', i+1)
    val_x = x_train[i * num_val_samples: (i+1) * num_val_samples]
    val_y = y_train[i * num_val_samples: (i+1) * num_val_samples]
    
    partial_train_x = np.concatenate([x_train[:i * num_val_samples], x_train[(i+1) *\
                                                num_val_samples:]],axis=0)
    partial_train_y = np.concatenate([y_train[:i * num_val_samples], y_train[(i+1) *\
                                                num_val_samples:]], axis=0)
    original_model = origin_model()
    original_hist = original_model.fit(partial_train_x,partial_train_y,\
                                       validation_data =(val_x, val_y), epochs = num_epochs,\
                                       batch_size = bat_size, verbose=0)
    original_train_loss = original_hist.history['mse']
    original_val_loss = original_hist.history['val_mse']
    all_train_loss.append(original_train_loss)
    all_val_loss.append(original_val_loss)

end = datetime.datetime.now()
print ("Time required for training:",end - start)

average_mse_train = [np.mean([x[i] for x in all_train_loss]) for i in range(num_epochs)]
average_mse_val = [np.mean([x[i] for x in all_val_loss]) for i in range(num_epochs)]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Visualize training loss history
plt.plot(range(1, len(average_mse_val) + 1), average_mse_val, label='Validation loss')
plt.plot(range(1, len(average_mse_train) + 1), average_mse_train, label='Training loss')
plt.title("Training vs. Validation MSE")
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend()
plt.show()

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mse_val = smooth_curve(average_mse_val[10:])
smooth_mse_train = smooth_curve(average_mse_train[10:])
plt.plot(range(1, len(smooth_mse_train) + 1), smooth_mse_train, label='Training loss')
plt.plot(range(1, len(smooth_mse_val) + 1), smooth_mse_val, label='Validation loss')
plt.title("Training vs. Validation MSE (excluding first 10 points)")
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# evaluation the model
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)

