import argparse
from data import DataFile, get_all_filenames, DATA_PREFIX
import joblib
import numpy as np
from keras.models import Sequential, save_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN
from sklearn.preprocessing import StandardScaler
from tools import re_order, plot


parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True,
                    choices=['Intra', 'Cross'], help='Mode to run the script in')
parser.add_argument('--downsample', type=int,
                    default=1, help='Downsample rate')
parser.add_argument('--model', required=False,
                    choices=['cnn', 'lstm', 'rnn'], default='cnn', help='Model to use')

args = parser.parse_args()
TAB_SIZE: int = 5
MODE: str = args.mode
DOWNSAMPLE: int = args.downsample
MODEL: str = args.model


def create_cnn(custom_input_shape: tuple):
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
              activation='relu',  input_shape=custom_input_shape))  # (X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_lstm(custom_input_shape: tuple):
    """
    Creates an LSTM model

    Returns
    -------
    Sequential The LSTM model

    TODO:
    - Parameterize 
    """
    model = Sequential()

    # LSTM layer with 50 units and 'relu' activation function
    model.add(LSTM(4, activation='relu', input_shape=custom_input_shape))

    # Output layer with the number of units equal to the number of classes
    model.add(Dense(units=4, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_simple_rnn(custom_input_shape: tuple):
    """
    Creates a simple RNN model.

    Returns
    -------
    Sequential The RNN model
    """
    model = Sequential()

    # Simple RNN layer with 50 units and 'relu' activation function
    model.add(SimpleRNN(50, activation='relu', input_shape=custom_input_shape))

    # Dense layer with 32 units and 'relu' activation function
    model.add(Dense(32, activation='relu'))

    # Output layer with the number of units equal to the number of classes
    model.add(Dense(units=4, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


root = f"{DATA_PREFIX}/{MODE}/train/"

# We re-order the files to have periodically appear when training
all_files = re_order(get_all_filenames(root))

scaler = StandardScaler()
custom_input_shape = (None, None)
data_files = []
X_train = []
y_train = []

for file_name in all_files:
    dat = DataFile(filename=file_name, scaler=scaler,
                   root_dir=root, downsample_rate=DOWNSAMPLE)
    data_files.append(dat)
    custom_input_shape = dat.get_matrix().shape

print(f"{'--'*TAB_SIZE} Created scaler {'--'*TAB_SIZE}")

for dat_file in data_files:
    # dat_file.normalize(scaler)
    original_data = dat_file.get_matrix()
    normalized_data = scaler.transform(original_data)
    X_train.append(normalized_data)
    y_train.append(dat_file.get_goal())

print(f"{'--'*TAB_SIZE} Data normalized {'--'*TAB_SIZE}")

joblib.dump(scaler, f"{MODE.lower()}_{DOWNSAMPLE}_scaler.pkl")
print(f"{'--'*TAB_SIZE} Scaler saved into a pickle file {'--'*TAB_SIZE}")

X_train = np.array(X_train)
y_train = np.array(y_train)

custom_input_shape = (X_train.shape[1], X_train.shape[2])
print(f"X_train shape: {custom_input_shape} {'--'*TAB_SIZE}")

# Create the smodel -> Be careful on the shape tuple
if MODEL == 'cnn':
    print(f"{'--'*TAB_SIZE} Created CNN model {'--'*TAB_SIZE}")
    model = create_cnn(custom_input_shape=custom_input_shape)
elif MODEL == 'lstm':
    print(f"{'--'*TAB_SIZE} Created LSTM model {'--'*TAB_SIZE}")
    model = create_lstm(custom_input_shape=custom_input_shape)
elif MODEL == 'rnn':
    print(f"{'--'*TAB_SIZE} Created RNN model {'--'*TAB_SIZE}")
    model = create_simple_rnn(custom_input_shape=custom_input_shape)

print(f"{'--'*TAB_SIZE} Model created {'--'*TAB_SIZE}")


X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

# Fit our model -> Might want to create a peridical fitting code
history = model.fit(X_train, y_train, epochs=1,
                    batch_size=1, validation_split=0.25)
print(f"{'--'*TAB_SIZE} Model fitted {'--'*TAB_SIZE}")


# Call the plot function to plot the loss and accuracy (as of now plots are shown, not saved)
plot(history.history, save=False, show=True, name=f"model")
print(f"{'--'*TAB_SIZE} Plots created and/or saved {'--'*TAB_SIZE}")

save_model(model, f"{MODE.lower()}_{DOWNSAMPLE}_{MODEL}_model.h5")
print(f"{'--'*TAB_SIZE} Model saved into a pickle file {'--'*TAB_SIZE}")
