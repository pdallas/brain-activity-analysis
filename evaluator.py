import numpy as np
from keras.models import load_model
from data import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True,
                    choices=['Intra', 'Cross'], help='Mode to run the script in')
parser.add_argument('--downsample', type=int,
                    default=1, help='Downsample rate')


args = parser.parse_args()
MODE: str = args.mode
DOWNSAMPLE: int = args.downsample

if MODE == "Intra":
    root = f"{DATA_PREFIX}/{MODE}/test/"
    X_test, y_test = [], []
    all_files = get_all_filenames(root)
    for file_name in all_files:
        dat = DataFile(filename=file_name, root_dir=root,
                       downsample_rate=DOWNSAMPLE)
        arr = dat.get_matrix()
        out = dat.goal_id
        X_test.append(arr)
        y_test.append(out)
        dat.remove()

elif MODE == "Cross":
    root = f"{DATA_PREFIX}/{MODE}/test1/"
    X_test, y_test = [], []
    all_files = get_all_filenames(root)
    for file_name in all_files:
        dat = DataFile(filename=file_name, root_dir=root,
                       downsample_rate=DOWNSAMPLE)
        arr = dat.get_matrix()
        out = dat.goal_id
        X_test.append(arr)
        y_test.append(out)
        dat.remove()

    root = f"{DATA_PREFIX}/{MODE}/test2/"
    all_files = get_all_filenames(root)
    for file_name in all_files:
        dat = DataFile(filename=file_name, root_dir=root,
                       downsample_rate=DOWNSAMPLE)
        arr = dat.get_matrix()
        out = dat.goal_id
        X_test.append(arr)
        y_test.append(out)
        dat.remove()

    root = f"{DATA_PREFIX}/{MODE}/test3/"
    all_files = get_all_filenames(root)
    for file_name in all_files:
        dat = DataFile(filename=file_name, root_dir=root,
                       downsample_rate=DOWNSAMPLE)
        arr = dat.get_matrix()
        out = dat.goal_id
        X_test.append(arr)
        y_test.append(out)
        dat.remove()

X_test = np.array(X_test)  # Convert to NumPy array
y_test = np.array(y_test)  # Convert to NumPy array
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Load the Keras model
model = load_model(f"{MODE.lower()}_{DOWNSAMPLE}_cnn_model.h5")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
