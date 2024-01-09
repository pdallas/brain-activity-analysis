from keras.models import save_model
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import pretty_errors
from data import *
from tools import *

DOWNSAMPLE = 1
EPOCHS = 10
BATCH_SIZE = 1
X_train, y_train = [], []

root = f"{DATA_PREFIX}/Intra/train/"
all_files = get_all_filenames(root)
all_files = re_order(all_files)  # We re order the files to avoid overfitting


for file_name in all_files:
    dat = DataFile(filename=file_name, root_dir=root,
                   downsample_rate=DOWNSAMPLE)
    arr = dat.get_matrix()
    out = dat.goal_id
    X_train.append(arr)
    y_train.append(out)
    dat.remove()

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])


def create_lstm(units, dropout=0.01):
    model = Sequential()
    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def create_cnn():
    """
    Creates a CNN model

    Returns
    -------
    Sequential The CNN model

    TODO:
    - Parameterize 
    """
    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=3,
              activation='relu',  input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = create_cnn()

history = model.fit(X_train, y_train, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, validation_split=0.25)


# Call the plot function to plot the loss and accuracy (as of now plots are shown, not saved)
plot(history.history, save=False, show=True, name=f"cnn_model")

save_model(model, f"cnn_model.h5")
