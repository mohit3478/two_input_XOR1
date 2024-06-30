import numpy as np
from tensorflow import keras
from src.preprocessing.data_management import load_model

def predict(input_data):
    model = load_model("two_input_xor_nn.h5")
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = predict(test_data)
    print("Predictions:", predictions)
