from data import *
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

TIMESTEP: int = 10
ROWS: int = 3563  # (X_train.shape[1], X_train.shape[2])
DOWNSAMPLE: int = 10
BATCH_SIZE: int = 32

# input_shape = (X_train.shape[1], X_train.shape[2])

check = False
model = Sequential()


def create_dataset(X, label, time_steps=1):
    Xs, ys = [], []
    for i in range(X.shape[1] - time_steps):
        v = X[:, i:(i + time_steps)].T
        Xs.append(v)
        ys.append(label)
    return np.array(Xs), np.array(ys)

HISTORY = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
root = f"{DATA_PREFIX}/Intra/train/"
all_files = get_all_filenames(root)

for file_name in all_files:
    dat = DataFile(filename=file_name, root_dir=root,
                   downsample_rate=DOWNSAMPLE)
    X, y = create_dataset(dat.get_matrix(), dat.goal_id, TIMESTEP)
    if not check:
        input_shape = (X.shape[1], X.shape[2])
        model.add(LSTM(32, input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'], run_eagerly=True)
        check = True

    history = model.fit(
        X, y, epochs=1, batch_size=BATCH_SIZE, validation_split=0.2)
    dat.remove()
    del X
    del y
    for key in HISTORY.keys():
        HISTORY[key].extend(history.history[key])

try:
    # Plot training & validation accuracy values
    plt.plot(HISTORY['accuracy'])
    plt.plot(HISTORY['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(HISTORY['loss'])
    plt.plot(HISTORY['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
except:
    print("Error plotting")

root = f"{DATA_PREFIX}/Intra/test/"
all_files = get_all_filenames(root)
for file_name in all_files:
    dat = DataFile(filename=file_name, root_dir=root, downsample_rate=10)
    X_test, y_test = create_dataset(dat.get_matrix(), dat.goal_id, TIMESTEP)
    loss, accuracy = model.evaluate(X_test, y_test)
    print("----" * 30)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print("----" * 30)
