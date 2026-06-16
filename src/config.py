import os
import tensorflow as tf

# Hardware safeguard to prevent thermal throttling
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

DATA_PATH = os.path.join("data", "diabetes_prediction_dataset.csv")
SAVE_DIR = "saved_models"
TARGET = "diabetes"
CLASSES = ["Negative", "Positive"]
NUM_CLASSES = 2

os.makedirs(SAVE_DIR, exist_ok=True)