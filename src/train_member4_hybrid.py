import time
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, Conv1D, MaxPooling1D, Reshape, Attention, GlobalAveragePooling1D
from src.config import SAVE_DIR, NUM_CLASSES
from src.preprocessing import load_and_preprocess_data
from src.utils import evaluate_and_save

def main():
    X_train, X_test, y_train, y_test, class_weights = load_and_preprocess_data()
    dim = X_train.shape[1]

    inputs = Input(shape=(dim,))
    x = Reshape((dim, 1))(inputs)
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    # Native Keras Attention Layer
    attn_out = Attention()([lstm_out, lstm_out])
    
    x = GlobalAveragePooling1D()(attn_out)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    start = time.time()
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32, class_weight=class_weights, verbose=1)
    train_time = time.time() - start
    
    model.save(os.path.join(SAVE_DIR, "member4_cnn_bilstm_attention.keras"))
    evaluate_and_save("member4_cnn_bilstm_attention", model, X_test, y_test, history, train_time)

if __name__ == "__main__":
    main()