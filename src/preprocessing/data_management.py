import os
import pandas as pd
import tensorflow as tf
from src.config import config

def load_dataset(file_name):
    file_path = os.path.join(config.DATA_PATH, file_name)
    dataset = pd.read_csv(file_path)
    return dataset

def save_model(model, file_name):
    model_path = os.path.join(config.SAVED_MODEL_PATH, file_name)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(file_name):
    model_path = os.path.join(config.SAVED_MODEL_PATH, file_name)
    model = tf.keras.models.load_model(model_path)
    return model
