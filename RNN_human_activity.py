"""
Math Modeling Final Project
Description: 
This file uses a simple recurrent neural network to predict human activity
using smartphone sensor data.

Keras, which wraps tensorflow, is the module used here. The data set
(found here: https://archive.ics.uci.edu/ml/datasets/Human+Activity+
Recognition+Using+Smartphones) provides data with 561 features derived from
a Samsung Galaxy SII's gyroscope and accelerometer. There are 6 activities in
the data set: walking (1), walking up stairs (2), walking down stairs (3),
sitting (4), standing (5), laying (6).  

In hindsight, I probably could have used a pandas dataframe to 
read the .txt file in as a csv (pandas.DataFrame.read_csv(<filepath>))
instead of writing my own function.

Created on Thu Mar 29 09:30:23 2018

author = Eric Gabriel
"""
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras import regularizers
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def txt_to_np(f_directory):
    """
    This function converts a .txt files for this project to a numpy array of
    arrays or numpy array. It assumes all values in the .txt file are floats.
    It was designed for this project's data set in particular, so there are no 
    guarantees that it works elsewhere.
    
    :param f_directory: directory of the .txt file from which to read
    :return data_array: numpy 2-D matrix of the .txt file or a numpy 1-D array
    if there is only 1 attribute (as in y_train and y_test)
    """
    with open(f_directory) as txt:
        # get num_lines for X_train's dimension --> avoid copying X_train over and over 
        num_lines = 0
        for row_num, line_str in enumerate(txt):
            num_lines = row_num
        
        num_attributes = len(line_str.split())  # number of columns
        data_array = np.ndarray(shape=(num_lines, num_attributes),
                                dtype='float') # num_lines by 561 attributes
        
        txt.seek(0)  # go back and start from beginning
        
        # for a row
        for row_num, line_str in enumerate(txt): 
            row_list = line_str.split() # split w/ whitespace as delimiter
            
            # for a column in this row
            for attrib_num, attrib_value in enumerate(row_list): 
                data_array[row_num - 1][attrib_num] = float(attrib_value)
              
        if num_attributes == 1:
            return np.ravel(data_array) # data_array[i][j] to just [i]
        else:
            return data_array
        
    return None   # return None if unsuccessful opening, reading, etc.


## Importing the data
root_dir = '/Users/EricG/Downloads/UCI HAR Dataset/'

# Get X_train array from working directory or read in .txt data
try: 
    X_train = np.load('X_train.npy')  # .npy = numpy specific file format
except:
    X_train_dir = root_dir + 'train/X_train.txt'
    X_train = txt_to_np(X_train_dir)    # read in 
    np.save('X_train.npy', X_train)     # save X_train for next time

# Get y_train array from working directory or read in .txt data
try:
    y_train = np.load('y_train.npy')
except:
    y_train_dir = root_dir + 'train/y_train.txt'
    y_train = txt_to_np(y_train_dir)    # read in 
    np.save('y_train.npy', y_train)     # save y_train array for next time
    
# Get X_test array from working directory or read in .txt data
try:
    X_test = np.load('X_test.npy')
except:
    X_test_dir = root_dir  + 'test/X_test.txt'
    X_test = txt_to_np(X_test_dir)  # read in 
    np.save('X_test.npy', X_test)   # save X_test matrix
    
# Get y_test array from working directory or read in .txt data
try:
    y_test = np.load('y_test.npy')
except:
    y_test_dir = root_dir  + 'test/y_test.txt'
    y_test = txt_to_np(y_test_dir)  # read in
    np.save('y_test.npy', y_test)   # save y_test matrix

# Preprocess the data - helps reduce training time and improves accuracy
lda = LinearDiscriminantAnalysis()
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Keras SimpleRNN expects an input of size:
# (num_training_examples, num_features, num_timesteps)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# size of batch on which model is trained
# if doing stateful RNNs, size_of_batch must be a factor of 
# X_train.shape[0] and X_test.shape[0]
# smaller batch size requires fewer epochs, but each epoch takes more time
size_of_batch = 128

#try test on batch etc

model = Sequential() # empty Sequential model; have to add layers 
model.add(SimpleRNN(10, activation='tanh',
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    kernel_regularizer=regularizers.l2(0.04),
                    return_sequences=True,
                    ))  # hidden layer 1
model.add(SimpleRNN(2, activation='tanh',
                    kernel_regularizer=regularizers.l2(0.0005)))
model.add(Dense(1, activation='elu'))  # output layer

model.compile(loss='mse',  # mean squared err. - measure precision and bias
              optimizer='nadam', # Nesterov adam optimzer
              metrics=['accuracy'])
 
model.fit(X_train, y_train, batch_size=size_of_batch, epochs=30,
          shuffle=False)

y_test_pred = model.predict(X_test,
                            batch_size=size_of_batch)

# predictions are not integers, so need to round them
y_test_pred_rounded = y_test_pred.round()

print(accuracy_score(y_test, y_test_pred_rounded))
print(confusion_matrix(y_test, y_test_pred_rounded))
