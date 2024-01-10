import argparse
from data import *
import joblib
from keras.models import load_model
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True,
                    choices=['Intra', 'Cross'], help='Mode to run the script in')
parser.add_argument('--downsample', type=int,
                    default=1, help='Downsample rate')


args = parser.parse_args()
MODE: str = args.mode
DOWNSAMPLE: int = args.downsample

# Load the scaler (created in cnn.py)
SCALER = joblib.load(f"{MODE.lower()}_{DOWNSAMPLE}_scaler.pkl")

if MODE == "Intra":
    root = f"{DATA_PREFIX}/{MODE}/test/"
    X_test, y_test = [], []
    all_files = get_all_filenames(root)
    for file_name in all_files:
        dat = DataFile(filename=file_name, root_dir=root,
                       downsample_rate=DOWNSAMPLE)

        original_data = dat.get_matrix()
        normalized_data = SCALER.transform(original_data)
        X_test.append(normalized_data)
        y_test.append(dat.get_goal())
        dat.remove()

elif MODE == "Cross":
    for test_label in ["test1", "test2", "test3"]:
        root = f"{DATA_PREFIX}/{MODE}/{test_label}/"
        X_test, y_test = [], []
        all_files = get_all_filenames(root)
        for file_name in all_files:
            dat = DataFile(filename=file_name, root_dir=root,
                           downsample_rate=DOWNSAMPLE)
            original_data = dat.get_matrix()
            normalized_data = SCALER.transform(original_data)
            X_test.append(normalized_data)
            y_test.append(dat.get_goal())
            dat.remove()

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Load the Keras model
model = load_model(f"{MODE.lower()}_{DOWNSAMPLE}_cnn_model.h5")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
