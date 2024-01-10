import argparse
from data import DataFile, get_all_filenames, DATA_PREFIX
import joblib
import numpy as np
from keras.models import Sequential, save_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import StandardScaler
from tools import re_order, plot


parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True,
                    choices=['Intra', 'Cross'], help='Mode to run the script in')
parser.add_argument('--downsample', type=int,
                    default=1, help='Downsample rate')


args = parser.parse_args()
TAB_SIZE: int = 5
MODE: str = args.mode
DOWNSAMPLE: int = args.downsample


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

# Create the CNN model -> Be careful on the shape tuple
cnn_model = create_cnn(custom_input_shape=custom_input_shape)
print(f"{'--'*TAB_SIZE} Model created {'--'*TAB_SIZE}")


X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

# Fit our model -> Might want to create a peridical fitting code
history = cnn_model.fit(X_train, y_train, epochs=10,
                        batch_size=1, validation_split=0.25)
print(f"{'--'*TAB_SIZE} Model fitted {'--'*TAB_SIZE}")


# Call the plot function to plot the loss and accuracy (as of now plots are shown, not saved)
plot(history.history, save=False, show=True, name=f"cnn_model")
print(f"{'--'*TAB_SIZE} Plots created and/or saved {'--'*TAB_SIZE}")

save_model(cnn_model, f"{MODE.lower()}_{DOWNSAMPLE}_cnn_model.h5")
print(f"{'--'*TAB_SIZE} Model saved into a pickle file {'--'*TAB_SIZE}")
