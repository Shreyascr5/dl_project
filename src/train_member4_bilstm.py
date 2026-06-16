import time
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, Reshape
from src.config import SAVE_DIR, NUM_CLASSES
from src.preprocessing import load_and_preprocess_data
from src.utils import evaluate_and_save

def main():
    X_train, X_test, y_train, y_test, class_weights = load_and_preprocess_data()
    dim = X_train.shape[1]

    model = Sequential([
        Reshape((1, dim), input_shape=(dim,)),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    start = time.time()
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32, class_weight=class_weights, verbose=1)
    train_time = time.time() - start
    
    model.save(os.path.join(SAVE_DIR, "member4_bilstm.keras"))
    evaluate_and_save("member4_bilstm", model, X_test, y_test, history, train_time)

if __name__ == "__main__":
    main()