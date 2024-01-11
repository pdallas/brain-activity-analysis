from keras.models import save_model
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
import pretty_errors
from data import *
from tools import *

import argparse

import keras_tuner as kt
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True,
                    choices=['Intra', 'Cross'], help='Mode to run the script in')
parser.add_argument('--downsample', type=int,
                    default=1, help='Downsample rate')


args = parser.parse_args()
MODE: str = args.mode
DOWNSAMPLE: int = args.downsample

EPOCHS = 10
BATCH_SIZE = 1
X_train, y_train = [], []

root = f"{DATA_PREFIX}/{MODE}/train/"
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


# def create_cnn():
#     """
#     Creates a CNN model

#     Returns
#     -------
#     Sequential The CNN model

#     TODO:
#     - Parameterize 
#     """
#     model = Sequential()

#     model.add(Conv1D(filters=32, kernel_size=3,
#               activation='relu',  input_shape=(X_train.shape[1], X_train.shape[2])))
#     model.add(MaxPooling1D(pool_size=2))

#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))

#     model.add(Flatten())

#     model.add(Dense(units=128, activation='relu'))
#     model.add(Dense(units=4, activation='softmax'))
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy', metrics=['accuracy'])

#     return model

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

    model.add(Conv1D(filters=96, kernel_size=3,
              activation='relu',  input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=160, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(units=288, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_cnn_tuning(hp):
    model = Sequential()

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=512, step=32)

    model.add(Conv1D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
        activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(
        filters=hp.Int('filters_2', min_value=64, max_value=256, step=32),
        kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
        activation='relu'
    ))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(units=hp_units_1, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

tuner = kt.Hyperband(
    create_cnn,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='cnn_tuning'
)

# Early stopping to prevent overfitting
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. Here are the optimal values:

- Number of units in the first dense layer: {best_hps.get('units_1')}
- Number of filters in the first Conv1D layer: {best_hps.get('filters_1')}
- Kernel size for the first Conv1D layer: {best_hps.get('kernel_size_1')}
- Number of filters in the second Conv1D layer: {best_hps.get('filters_2')}
- Kernel size for the second Conv1D layer: {best_hps.get('kernel_size_2')}
- Learning rate for the optimizer: {best_hps.get('learning_rate')}
""")


# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
#model = create_cnn()

history = model.fit(X_train, y_train, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, validation_split=0.25)


# Call the plot function to plot the loss and accuracy (as of now plots are shown, not saved)
plot(history.history, save=False, show=True, name=f"cnn_model")

#save_model(model, f"{MODE.lower()}_{DOWNSAMPLE}_cnn_model.h5")
