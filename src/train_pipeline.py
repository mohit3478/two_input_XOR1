import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from src.config import config
from src.preprocessing.data_management import load_dataset, save_model

def build_model():
    model = keras.Sequential([
        layers.Dense(2, input_dim=2, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def run_training():
    # Load the dataset
    training_data = load_dataset("train.csv")

    # Prepare the data
    X_train = training_data.iloc[:, 0:2].values
    Y_train = training_data.iloc[:, 2].values

    # Define the model
    model = build_model()

    # Train the model
    model.fit(X_train, Y_train, epochs=1000, batch_size=2, verbose=1)

    # Save the model
    save_model(model, "two_input_xor_nn.h5")

if __name__ == "__main__":
    run_training()
