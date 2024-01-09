import numpy as np
from keras.models import load_model
from data import *


root = f"{DATA_PREFIX}/Intra/test/"
X_test, y_test = [], []
all_files = get_all_filenames(root)
for file_name in all_files:
    dat = DataFile(filename=file_name, root_dir=root,
                   downsample_rate=1)
    arr = dat.get_matrix()
    out = dat.goal_id
    X_test.append(arr)
    y_test.append(out)
    dat.remove()

X_test = np.array(X_test)  # Convert to NumPy array
y_test = np.array(y_test)  # Convert to NumPy array
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Load the Keras model
model = load_model('cnn_model.h5')

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
