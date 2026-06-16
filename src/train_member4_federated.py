import os
import time
import random
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ✅ macOS fix
import matplotlib.pyplot as plt

# ==========================================================
# APPLE SILICON OPTIMIZATION FOR 16GB RAM
# ==========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Prevent OOM
os.environ["TENSORFLOW_ENABLE_ONEDNN_OPTS"] = "0"  # macOS compatibility

import tensorflow as tf

cpu_count = multiprocessing.cpu_count()

# ✅ CRITICAL: Reduced thread count for Apple Silicon M4
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(2)

# ✅ Reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# ==========================================================
# DATA LOADING & PREPROCESSING
# ==========================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Hospital names for simulation
HOSPITAL_NAMES = ["Hospital A", "Hospital B", "Hospital C", "Hospital D"]


def load_and_preprocess_data():
    """
    Load diabetes dataset and return train/test splits with class weights.
    Maintains consistency with centralized training preprocessing.
    """
    # ✅ Correct dataset path
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "diabetes_prediction_dataset.csv")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)

    # Robust target column detection
    target_col = None
    for col in df.columns:
        if col.lower() in ["outcome", "diabetes", "class"]:
            target_col = col
            break

    if target_col is None:
        raise ValueError("Target column not found. Available columns: " + str(df.columns))

    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Compute class weights for imbalance
    classes = np.unique(y_train)
    class_weights = dict(
        zip(
            classes,
            compute_class_weight("balanced", classes=classes, y=y_train),
        )
    )

    return X_train, X_test, y_train, y_test, class_weights


def find_best_member4_model():
    """
    Detect and return the best-performing Member 4 model.
    Checks for:
    - saved_models/member4_bilstm.keras
    - saved_models/member4_cnn_bilstm_attention.keras
    
    Returns path to the best model found, or raises error if none exist.
    """
    model_dir = Path("saved_models")
    
    if not model_dir.exists():
        raise FileNotFoundError(
            f"saved_models directory not found. "
            "Please ensure Member 4 models are in saved_models/ directory."
        )

    candidates = [
        model_dir / "member4_bilstm.keras",
        model_dir / "member4_cnn_bilstm_attention.keras",
    ]

    existing_models = [m for m in candidates if m.exists()]

    if not existing_models:
        raise FileNotFoundError(
            f"No Member 4 models found in {model_dir}. "
            "Expected one of: member4_bilstm.keras, member4_cnn_bilstm_attention.keras"
        )

    # Use the first available model (prioritize BiLSTM)
    best_model_path = existing_models[0]
    print(f"✅ Loading Member 4 model: {best_model_path}")
    return str(best_model_path)


def load_member4_model(model_path):
    """
    Load pre-trained Member 4 model.
    Preserves optimizer, loss, and metrics configuration.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Successfully loaded model from {model_path}")
        print(f"   Model summary:")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")


def reshape_for_recurrent(X):
    """
    Reshape data for recurrent architectures (LSTM, BiLSTM).
    Converts from (samples, features) to (samples, features, 1) for temporal dimension.
    
    Args:
        X: Array of shape (samples, features)
    
    Returns:
        Reshaped array of shape (samples, features, 1)
    """
    if len(X.shape) == 2:
        return X.reshape((X.shape[0], X.shape[1], 1))
    return X


def partition_federated_clients(X_train, y_train, num_clients=4):
    """
    Partition training data into non-overlapping federated clients.
    
    Args:
        X_train: Training features
        y_train: Training labels
        num_clients: Number of clients (default: 4 hospitals)
    
    Returns:
        List of tuples: [(X_client, y_client), ...]
    """
    chunk_size = len(X_train) // num_clients
    clients = []
    
    for i in range(num_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_clients - 1 else len(X_train)
        clients.append((X_train[start:end], y_train[start:end]))
        print(f"   📋 {HOSPITAL_NAMES[i]}: {end - start} samples")
    
    return clients


def federated_averaging(client_weights, client_sizes):
    """
    Implement Federated Averaging (FedAvg) aggregation.
    
    Weighted average of client weights based on their dataset sizes:
    w_global = (1/N) * sum(n_k / N * w_k)
    
    where:
    - N: total samples across all clients
    - n_k: samples in client k
    - w_k: weights from client k
    
    Args:
        client_weights: List of weight arrays from each client
        client_sizes: List of dataset sizes for each client
    
    Returns:
        Aggregated weight arrays
    """
    total_samples = sum(client_sizes)
    aggregated_weights = []

    for layer_weights in zip(*client_weights):
        weighted_layer = np.zeros_like(layer_weights[0])
        for idx in range(len(layer_weights)):
            weight_fraction = client_sizes[idx] / total_samples
            weighted_layer += layer_weights[idx] * weight_fraction
        aggregated_weights.append(weighted_layer)

    return aggregated_weights


def evaluate_and_save(model_name, model, X_test, y_test, train_time):
    """
    Evaluate model on test set and save metrics to CSV.
    
    Args:
        model_name: Name for the saved metrics file
        model: Trained model
        X_test: Test features (reshaped if needed)
        y_test: Test labels
        train_time: Training duration in seconds
    """
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Final Evaluation: Loss={loss:.4f}, Accuracy={acc:.4f}")

    # Save metrics
    results_path = os.path.join(SAVE_DIR, f"{model_name}_metrics.csv")
    pd.DataFrame({
        "Loss": [loss],
        "Accuracy": [acc],
        "TrainTimeSec": [int(train_time)],
    }).to_csv(results_path, index=False)

    print(f"💾 Metrics saved to {results_path}")


def save_federated_progression(global_accuracy_history, rounds):
    """
    Generate and save federated learning progression plot.
    
    Args:
        global_accuracy_history: List of accuracy values per round
        rounds: Total number of rounds
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, rounds + 1),
        global_accuracy_history,
        marker="o",
        linewidth=2,
        label="Global Model Accuracy",
    )
    plt.title("Federated Learning Accuracy Across Communication Rounds")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, rounds + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(SAVE_DIR, "federated_progression.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Progression plot saved to {plot_path}")


def save_federated_metrics(global_accuracy_history, rounds):
    """
    Save federated results to CSV.
    
    Args:
        global_accuracy_history: List of accuracy values per round
        rounds: Total number of rounds
    """
    federated_results = pd.DataFrame({
        "Round": list(range(1, rounds + 1)),
        "Global Accuracy": global_accuracy_history,
    })

    results_path = os.path.join(SAVE_DIR, "federated_results.csv")
    federated_results.to_csv(results_path, index=False)
    print(f"📈 Federated results saved to {results_path}")


# ==========================================================
# MAIN FEDERATED LEARNING SIMULATION
# ==========================================================

def main():
    print("\n" + "=" * 70)
    print("🚀 FEDERATED LEARNING SIMULATION - DIABETES PREDICTION")
    print("🍎 OPTIMIZED FOR APPLE SILICON M4 (16GB RAM)")
    print("=" * 70)

    # ======================================================
    # LOAD DATA
    # ======================================================

    print("\n📂 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, class_weights = load_and_preprocess_data()

    # ✅ REDUCED for Apple Silicon: Cap at 8000 samples for low memory usage
    MAX_FL_SAMPLES = min(8000, len(X_train))
    X_train = X_train[:MAX_FL_SAMPLES]
    y_train = y_train[:MAX_FL_SAMPLES]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   ⚠️  Reduced for 16GB MacBook Air M4 compatibility")

    # ======================================================
    # LOAD BEST MEMBER 4 MODEL
    # ======================================================

    print("\n🤖 Loading pre-trained Member 4 model...")
    model_path = find_best_member4_model()
    global_model = load_member4_model(model_path)

    # Reshape data for recurrent architectures
    X_train_fed = reshape_for_recurrent(X_train)
    X_test_fed = reshape_for_recurrent(X_test)

    print(f"   Reshaped training data: {X_train_fed.shape}")
    print(f"   Reshaped test data: {X_test_fed.shape}")

    # ======================================================
    # PARTITION DATA INTO FEDERATED CLIENTS
    # ======================================================

    print("\n🏥 Partitioning data into federated clients...")
    hospitals = partition_federated_clients(X_train_fed, y_train, num_clients=4)

    global_weights = global_model.get_weights()

    # ======================================================
    # FEDERATED LEARNING CONFIGURATION
    # ✅ OPTIMIZED FOR 16GB MacBook Air M4
    # ======================================================

    ROUNDS = 5  # ✅ REDUCED from 10 to 5
    LOCAL_EPOCHS = 1  # ✅ REDUCED from 2 to 1
    CLIENTS_PER_ROUND = 2  # ✅ REDUCED from 4 to 2 (sequential training)
    BATCH_SIZE = 16  # ✅ REDUCED from 32 to 16

    print(f"\n⚙️  Federated Learning Configuration:")
    print(f"   Rounds: {ROUNDS}")
    print(f"   Clients per round: {CLIENTS_PER_ROUND}")
    print(f"   Local epochs: {LOCAL_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   ⚠️  Reduced for 16GB MacBook Air M4 compatibility")

    global_accuracy_history = []
    start_time = time.time()

    # ======================================================
    # FEDERATED TRAINING LOOP
    # ======================================================

    for round_idx in range(ROUNDS):
        print("\n" + "=" * 70)
        print(f"🌐 FEDERATED ROUND {round_idx + 1}/{ROUNDS}")
        print("=" * 70)

        client_weights = []
        client_sizes = []
        selected_clients = list(range(CLIENTS_PER_ROUND))

        # --- Client Training Phase ---
        for client_id in selected_clients:
            hospital_name = HOSPITAL_NAMES[client_id]
            X_client, y_client = hospitals[client_id]

            print(f"\n   🏥 {hospital_name} ({len(X_client)} samples)")
            print(f"      └─ Class distribution: {np.bincount(y_client.astype(int))}")

            # Set current global weights
            global_model.set_weights(global_weights)

            # ✅ Local training with reduced verbosity
            try:
                history = global_model.fit(
                    X_client,
                    y_client,
                    epochs=LOCAL_EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                    class_weight=class_weights,
                    shuffle=True,
                )

                # Collect updated weights and dataset size
                client_weights.append(global_model.get_weights())
                client_sizes.append(len(X_client))

                # Log training result
                final_loss = history.history["loss"][-1]
                final_acc = history.history["accuracy"][-1]
                print(f"      └─ Local Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
            except Exception as e:
                print(f"      ⚠️  Error training {hospital_name}: {str(e)}")
                print(f"      └─ Skipping this client's update")
                continue

        if not client_weights:
            print("⚠️  No clients trained successfully. Skipping aggregation.")
            continue

        # --- FedAvg Aggregation Phase ---
        print("\n   🔄 Aggregating weights via FedAvg...")
        aggregated_weights = federated_averaging(client_weights, client_sizes)
        global_weights = aggregated_weights
        global_model.set_weights(global_weights)

        # --- Global Evaluation ---
        try:
            loss, accuracy = global_model.evaluate(X_test_fed, y_test, verbose=0)
            global_accuracy_history.append(accuracy)
            print(f"   ✅ Global Model - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"   ⚠️  Error evaluating model: {str(e)}")

    print("\n" + "=" * 70)
    print("🎯 FEDERATED TRAINING COMPLETED")
    print("=" * 70)

    # ======================================================
    # SAVE FINAL FEDERATED MODEL
    # ======================================================

    total_training_time = time.time() - start_time
    federated_model_path = os.path.join(SAVE_DIR, "federated_global_model.keras")
    
    try:
        global_model.save(federated_model_path)
        print(f"\n💾 Federated global model saved: {federated_model_path}")
    except Exception as e:
        print(f"\n⚠️  Error saving model: {str(e)}")
        return

    # ======================================================
    # GENERATE OUTPUTS
    # ======================================================

    print("\n📊 Generating outputs...")

    # Save metrics
    try:
        evaluate_and_save(
            model_name="federated_global_model",
            model=global_model,
            X_test=X_test_fed,
            y_test=y_test,
            train_time=total_training_time,
        )
    except Exception as e:
        print(f"⚠️  Error saving metrics: {str(e)}")

    # Save progression plot
    if global_accuracy_history:
        try:
            save_federated_progression(global_accuracy_history, len(global_accuracy_history))
        except Exception as e:
            print(f"⚠️  Error saving progression plot: {str(e)}")

    # Save federated results CSV
    if global_accuracy_history:
        try:
            save_federated_metrics(global_accuracy_history, len(global_accuracy_history))
        except Exception as e:
            print(f"⚠️  Error saving federated metrics: {str(e)}")

    # ======================================================
    # SUMMARY
    # ======================================================

    print("\n" + "=" * 70)
    print("🎉 FEDERATED LEARNING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • Communication Rounds    : {ROUNDS}")
    print(f"  • Hospitals (Clients)     : {CLIENTS_PER_ROUND}")
    print(f"  • Local Epochs            : {LOCAL_EPOCHS}")
    print(f"  • Batch Size              : {BATCH_SIZE}")
    print(f"  • Training Samples        : {len(X_train)}")
    print(f"  • Training Time           : {int(total_training_time)}s")

    print(f"\nGenerated Files:")
    print(f"  ✔ {os.path.join(SAVE_DIR, 'federated_global_model.keras')}")
    print(f"  ✔ {os.path.join(SAVE_DIR, 'federated_global_model_metrics.csv')}")
    print(f"  ✔ {os.path.join(SAVE_DIR, 'federated_results.csv')}")
    print(f"  ✔ {os.path.join(SAVE_DIR, 'federated_progression.png')}")

    print(f"\n✨ Model ready for Streamlit dashboard integration!")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
