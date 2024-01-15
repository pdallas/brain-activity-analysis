import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
parser.add_argument('--model', required=False,
                    choices=['cnn', 'lstm', 'rnn'], default='cnn', help='Model to use')

args = parser.parse_args()
MODE: str = args.mode
DOWNSAMPLE: int = args.downsample
MODEL: str = args.model
TAB_SIZE: int = 5

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
model = load_model(f"{MODE.lower()}_{DOWNSAMPLE}_{MODEL}_model.h5")
print(f"{'--'*TAB_SIZE} {MODEL.upper()} model loaded {'--'*TAB_SIZE}")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# create a confusion matrix

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
# map the labels to the corresponding activities
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 10))
labels = ['Resting', 'Math & Story', 'Working Memory', 'Motor']
sns.heatmap(cm, annot=True, square=True, cmap=plt.cm.Blues,
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title(f"{MODE} {MODEL.upper()} Confusion Matrix")
plt.savefig(f"{MODE.lower()}_{DOWNSAMPLE}_{MODEL}_confusion_matrix.png")
plt.show()
