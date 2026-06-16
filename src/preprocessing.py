import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from src.config import DATA_PATH, TARGET, SAVE_DIR
from sklearn.utils.class_weight import compute_class_weight

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    
    # 1. Cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    # 2. Categorical Encoding (get_dummies prevents 'Female' string errors)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
        
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])
    
    # Validation
    if len(X.select_dtypes(exclude=["number"]).columns) > 0:
        raise ValueError("Non-numeric columns remain!")

    # 3. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Class Weights (Diabetes dataset is heavily imbalanced)
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, class_weights

def get_federated_clients(X_train, y_train, num_clients=4):
    """Splits data into isolated Hospital A, B, C, D."""
    client_data = []
    chunk_size = len(X_train) // num_clients
    for i in range(num_clients):
        start = i * chunk_size
        end = start + chunk_size if i < num_clients - 1 else len(X_train)
        client_data.append((X_train[start:end], y_train[start:end]))
    return client_data