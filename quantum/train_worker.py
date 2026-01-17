#!/usr/bin/env python3
"""
Quantum Variational Circuit Training Worker

This script trains an individual quantum classifier model for wildfire prediction
using PennyLane's variational quantum circuit (VQC) framework. The model uses
amplitude embedding to encode classical features into quantum states, followed by
parameterized rotation gates and entangling operations to learn classification
boundaries.

Key Features:
    - Amplitude embedding for efficient quantum state preparation
    - Configurable circuit depth (2-3 layers) with diverse hyperparameters
    - Class-weighted loss function to handle severe imbalance (~5% fires)
    - Optional SMOTE oversampling for minority class augmentation
    - Early stopping with patience to prevent overfitting
    - Test prediction caching in final session for fast evaluation
    - Session-based training for memory-efficient long runs

Architecture:
    Input: 64 classical features → Amplitude embedding → 6 data qubits
    Processing: 6 additional qubits for increased expressivity
    Circuit: RY/RZ rotations + CNOT entanglement (2-3 layers)
    Output: Single expectation value on qubit 0 (binary classification)

Usage:
    python train_worker.py <model_id> <session> [OPTIONS]
    
    model_id: Integer 1-12 identifying which model configuration to use
    session: Integer 1-4 identifying current training session
    
    Optional arguments:
    --data PATH           : Path to preprocessed data shard (default: ./preprocessed/model_{id}.pkl)
    --checkpoint-dir DIR  : Checkpoint output directory (default: ./checkpoints)
    --samples N           : Samples per session (default: 5000)
    --iterations N        : Max iterations per session (default: 30)
    --threads N           : OpenMP threads (default: 4)

Dependencies:
    - pennylane: Quantum circuit simulation framework
    - numpy: Numerical operations (PennyLane's autograd-compatible version)
    - imblearn: SMOTE oversampling (optional, graceful fallback if missing)
"""

import sys
import os
import time
import pickle
import traceback
import argparse
from datetime import datetime

# ============================================================================
# MEMORY PROFILING UTILITY
# ============================================================================

try:
    import psutil
except Exception:
    psutil = None

def get_mem_gb():
    """Returns current process memory usage in gigabytes."""
    if psutil:
        return psutil.Process().memory_info().rss / 1e9
    else:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        except Exception:
            return -1.0

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command-line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description='Train quantum VQC model for wildfire prediction')
    
    # Required arguments
    parser.add_argument('model_id', type=int, help='Model ID (1-12)')
    parser.add_argument('session', type=int, help='Training session (1-4)')
    
    # Optional arguments
    parser.add_argument('--data', type=str, default=None,
                       help='Path to preprocessed data (default: ./preprocessed/model_<id>.pkl)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint output directory')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Samples per training session')
    parser.add_argument('--iterations', type=int, default=30,
                       help='Max gradient descent iterations per session')
    parser.add_argument('--threads', type=int, default=4,
                       help='OpenMP thread count')
    parser.add_argument('--use-smote', action='store_true',
                       help='Enable SMOTE oversampling')
    
    args = parser.parse_args()
    
    # Set default data path if not provided
    if args.data is None:
        args.data = f'./preprocessed/model_{args.model_id}.pkl'
    
    # Validate arguments
    if not 1 <= args.model_id <= 12:
        parser.error('model_id must be between 1 and 12')
    if not 1 <= args.session <= 4:
        parser.error('session must be between 1 and 4')
    
    return args

# Parse arguments
args = parse_args()
MODEL_ID = args.model_id
CURRENT_SESSION = args.session
DATA_PATH = args.data
CHECKPOINT_DIR = args.checkpoint_dir
SAMPLES_PER_SESSION = args.samples
MAX_ITERATIONS = args.iterations
USE_SMOTE = args.use_smote

# Set OpenMP threads for parallel circuit simulation
os.environ["OMP_NUM_THREADS"] = str(args.threads)

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print(f"QUANTUM VQC WORKER - MODEL {MODEL_ID} - SESSION {CURRENT_SESSION}")
print("=" * 70)
print(f"PID: {os.getpid()}")
print(f"Data: {DATA_PATH}")
print(f"Checkpoints: {CHECKPOINT_DIR}")
print(f"Samples/session: {SAMPLES_PER_SESSION}")
print(f"Max iterations: {MAX_ITERATIONS}")
print(f"OMP threads: {os.environ.get('OMP_NUM_THREADS')}")
print(f"Starting time: {datetime.now().isoformat()}")
print(f"[MEM] At start: {get_mem_gb():.3f} GB")
print()

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

N_MODELS = 12                    # Total number of parallel models in ensemble
N_SESSIONS = 4                   # Number of training sessions
N_QUBITS = 6                     # Data qubits for amplitude embedding
N_PROCESSING_QUBITS = 6          # Additional qubits for circuit expressivity
TOTAL_QUBITS = N_QUBITS + N_PROCESSING_QUBITS
PATIENCE = 5                     # Early stopping patience

# SMOTE configuration
SMOTE_TARGET_RATIO = 0.3         # Target minority class ratio after SMOTE

# ============================================================================
# MODEL-SPECIFIC HYPERPARAMETER CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    1:  {'learning_rate': 0.01,  'n_layers': 2, 'init_scale': 2 * 3.14159},
    2:  {'learning_rate': 0.02,  'n_layers': 2, 'init_scale': 3.14159},
    3:  {'learning_rate': 0.015, 'n_layers': 3, 'init_scale': 1.5 * 3.14159},
    4:  {'learning_rate': 0.005, 'n_layers': 2, 'init_scale': 0.5 * 3.14159},
    5:  {'learning_rate': 0.01,  'n_layers': 3, 'init_scale': 2.5 * 3.14159},
    6:  {'learning_rate': 0.025, 'n_layers': 2, 'init_scale': 1.2 * 3.14159},
    7:  {'learning_rate': 0.008, 'n_layers': 3, 'init_scale': 1.8 * 3.14159},
    8:  {'learning_rate': 0.012, 'n_layers': 2, 'init_scale': 0.8 * 3.14159},
    9:  {'learning_rate': 0.018, 'n_layers': 3, 'init_scale': 2.2 * 3.14159},
    10: {'learning_rate': 0.015, 'n_layers': 2, 'init_scale': 1.0 * 3.14159},
    11: {'learning_rate': 0.01,  'n_layers': 3, 'init_scale': 1.5 * 3.14159},
    12: {'learning_rate': 0.02,  'n_layers': 2, 'init_scale': 0.7 * 3.14159}
}

config = MODEL_CONFIGS.get(MODEL_ID, MODEL_CONFIGS[1])
N_LAYERS = config['n_layers']

print(f"Model config: LR={config['learning_rate']}, Layers={N_LAYERS}, Init={config['init_scale']:.2f}")
print()

# ============================================================================
# DEPENDENCY IMPORTS
# ============================================================================

try:
    import pennylane as qml
    from pennylane import numpy as np  # PennyLane's autograd-compatible numpy
except Exception as e:
    print("[ERROR] Failed to import PennyLane:", e)
    traceback.print_exc()
    sys.exit(1)

# Try to import SMOTE (optional)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    if USE_SMOTE:
        print("[WARN] SMOTE requested but imbalanced-learn not installed")
        print("[WARN] Continuing without SMOTE...")
        USE_SMOTE = False

# ============================================================================
# DATA LOADING
# ============================================================================

try:
    t0 = time.time()
    
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        print("Run preprocessing first: python preprocess.py")
        sys.exit(1)
    
    with open(DATA_PATH, 'rb') as f:
        d = pickle.load(f)
    
    # Load training data
    X_train = d['X_train'].astype(np.float32)
    y_train = d['y_train'].astype(int)
    
    # Load test data for final session caching
    X_test = d['X_test'].astype(np.float32)
    y_test = d['y_test'].astype(int)
    
    val_size = len(d['X_val'])
    
    print(f"[Model {MODEL_ID}] Loaded data from {DATA_PATH} in {time.time() - t0:.2f}s")
    print(f"[Model {MODEL_ID}] Train: {len(X_train)}, Val: {val_size}, Test: {len(X_test)}")
    print(f"[Model {MODEL_ID}] Original class balance: {y_train.mean():.3%} fires")
    print(f"[Model {MODEL_ID}] Memory after load: {get_mem_gb():.3f} GB")
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SESSION DATA PREPARATION
# ============================================================================

original_fire_pct = y_train.mean()

# Select subset for this session
start_idx = (CURRENT_SESSION - 1) * SAMPLES_PER_SESSION
end_idx = min(start_idx + SAMPLES_PER_SESSION, len(X_train))
X_session = X_train[start_idx:end_idx]
y_session = y_train[start_idx:end_idx]

print(f"\n[Model {MODEL_ID}] Session {CURRENT_SESSION}/{N_SESSIONS}")
print(f"[Model {MODEL_ID}] Samples: {len(X_session)} (indices {start_idx} to {end_idx})")
print(f"[Model {MODEL_ID}] Fire rate: {y_session.mean():.3%}")

# Apply SMOTE if enabled and needed
if USE_SMOTE and SMOTE_AVAILABLE and y_session.mean() < SMOTE_TARGET_RATIO:
    try:
        smote = SMOTE(sampling_strategy=SMOTE_TARGET_RATIO, random_state=42)
        X_session, y_session = smote.fit_resample(X_session, y_session)
        print(f"[Model {MODEL_ID}] SMOTE applied: {len(X_session)} samples, {y_session.mean():.3%} fires")
    except Exception as e:
        print(f"[WARN] SMOTE failed: {e}")

print(f"[Model {MODEL_ID}] Memory after session prep: {get_mem_gb():.3f} GB")

# ============================================================================
# QUANTUM CIRCUIT DEFINITION
# ============================================================================

dev = qml.device('lightning.qubit', wires=TOTAL_QUBITS)

@qml.qnode(dev)
def circuit(weights, x):
    """
    Variational quantum circuit for binary classification.
    
    Architecture:
        1. L2 normalization of input
        2. Amplitude embedding into 6 data qubits
        3. N_LAYERS repetitions of:
           - RY/RZ rotations on all 12 qubits
           - CNOT entanglement chain
        4. Measurement of expectation value on qubit 0
    
    Args:
        weights: Parameters of shape (n_layers, total_qubits, 2)
        x: Feature vector of length 64
    
    Returns:
        Expectation value in range [-1, 1]
    """
    # L2 normalize input
    x_norm = x / (np.linalg.norm(x) + 1e-8)
    
    # Amplitude embedding
    qml.AmplitudeEmbedding(x_norm, wires=range(N_QUBITS), normalize=True)
    
    # Parameterized layers
    for layer in range(N_LAYERS):
        # Rotation gates
        for qubit in range(TOTAL_QUBITS):
            qml.RY(weights[layer, qubit, 0], wires=qubit)
            qml.RZ(weights[layer, qubit, 1], wires=qubit)
        
        # Entanglement
        for qubit in range(TOTAL_QUBITS - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
    
    # Measure qubit 0
    return qml.expval(qml.PauliZ(0))

print(f"[Model {MODEL_ID}] Circuit: {N_LAYERS} layers, {TOTAL_QUBITS} qubits")
print(f"[Model {MODEL_ID}] Parameters: {N_LAYERS * TOTAL_QUBITS * 2}")

# ============================================================================
# PARAMETER INITIALIZATION
# ============================================================================

# Initialize weights
weights_shape = (N_LAYERS, TOTAL_QUBITS, 2)
weights = np.random.uniform(-config['init_scale'], config['init_scale'], weights_shape)

# Create optimizer
opt = qml.GradientDescentOptimizer(stepsize=config['learning_rate'])

print(f"[Model {MODEL_ID}] Optimizer: GradientDescent, LR={config['learning_rate']}")
print()

# ============================================================================
# TRAINING LOOP
# ============================================================================

try:
    session_start_time = time.time()
    cumulative_samples = 0
    
    # Training history
    history = {
        'iteration': [],
        'loss': [],
        'accuracy': [],
        'time': [],
        'samples_per_sec': []
    }
    
    # Early stopping
    best_loss = float('inf')
    best_weights = weights.copy()
    patience_counter = 0
    
    # Check for previous session checkpoint
    if CURRENT_SESSION > 1:
        prev_ckpt = os.path.join(CHECKPOINT_DIR, f'model_{MODEL_ID}_session_{CURRENT_SESSION-1}.pkl')
        if os.path.exists(prev_ckpt):
            try:
                with open(prev_ckpt, 'rb') as f:
                    prev_checkpoint = pickle.load(f)
                weights = prev_checkpoint['weights']
                print(f"[Model {MODEL_ID}] Loaded weights from previous session: {prev_ckpt}")
            except Exception as e:
                print(f"[WARN] Could not load previous checkpoint: {e}")
    else:
        print(f"\n[Model {MODEL_ID}] Starting training...")
        print("=" * 70)
    
    # Training iterations
    for iteration in range(MAX_ITERATIONS):
        iter_start = time.time()
        sys.stdout.flush()

        # Forward pass
        preds = np.array([circuit(weights, x) for x in X_session])
        y_pred = (preds > 0).astype(int)

        # Class-weighted loss
        n_samples = len(y_session)
        n_pos = y_session.sum()
        n_neg = n_samples - n_pos
        
        if n_pos == 0 or n_neg == 0:
            sample_weights = np.ones(n_samples)
        else:
            weight_for_0 = n_samples / (2 * n_neg)
            weight_for_1 = n_samples / (2 * n_pos)
            sample_weights = np.where(y_session == 1, weight_for_1, weight_for_0)
        
        # Weighted MSE loss
        loss = np.mean(sample_weights * (preds - (2 * y_session - 1)) ** 2)
        acc = np.mean(y_pred == y_session)

        # Gradient descent
        def cost_fn(w):
            p = np.array([circuit(w, x) for x in X_session])
            return np.mean(sample_weights * (p - (2 * y_session - 1)) ** 2)

        weights = opt.step(cost_fn, weights)

        # Metrics
        iter_time = time.time() - iter_start
        samples_per_sec = len(X_session) / iter_time if iter_time > 0 else 0
        cumulative_samples += len(X_session)
        elapsed_time = time.time() - session_start_time
        avg_throughput = cumulative_samples / elapsed_time if elapsed_time > 0 else 0
        
        # Record history
        history['iteration'].append(iteration + 1)
        history['loss'].append(float(loss))
        history['accuracy'].append(float(acc))
        history['time'].append(iter_time)
        history['samples_per_sec'].append(samples_per_sec)

        # Print progress
        print(f"  [M{MODEL_ID}] Iter {iteration + 1}/{MAX_ITERATIONS}: "
              f"Loss={loss:.6f}, Acc={acc:.4f}, "
              f"Time={iter_time:.2f}s, Samples/s={samples_per_sec:.1f}")
        print(f"         Cumulative: {cumulative_samples} samples, "
              f"Avg throughput={avg_throughput:.1f} samples/s, "
              f"Mem={get_mem_gb():.3f}GB")
        sys.stdout.flush()

        # Early stopping
        if loss < best_loss:
            best_loss = loss
            best_weights = weights.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [M{MODEL_ID}] Early stopping at iteration {iteration + 1}")
                weights = best_weights
                break

    # ========================================================================
    # TEST PREDICTION CACHING (FINAL SESSION ONLY)
    # ========================================================================
    
    test_predictions = None
    if CURRENT_SESSION == N_SESSIONS:
        print("=" * 70)
        print(f"[Model {MODEL_ID}] Final session - generating test predictions...")
        print(f"[Model {MODEL_ID}] Test set size: {len(X_test)} samples")
        pred_start = time.time()
        
        # Generate predictions in batches
        batch_size = 100
        test_predictions_raw = []
        
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            batch_preds = np.array([circuit(weights, x) for x in X_test[i:batch_end]])
            test_predictions_raw.extend(batch_preds)
            
            if (i // batch_size) % 5 == 0:
                elapsed = time.time() - pred_start
                progress = batch_end / len(X_test)
                eta = (elapsed / progress - elapsed) if progress > 0 else 0
                print(f"  Progress: {batch_end}/{len(X_test)} ({progress*100:.1f}%) - ETA: {eta:.1f}s")
                sys.stdout.flush()
        
        test_predictions_raw = np.array(test_predictions_raw)
        test_predictions = (test_predictions_raw > 0).astype(int)
        
        pred_time = time.time() - pred_start
        print(f"[Model {MODEL_ID}] Test predictions generated in {pred_time:.2f}s")
        print(f"[Model {MODEL_ID}] Throughput: {len(X_test)/pred_time:.1f} samples/sec")
        print(f"[Model {MODEL_ID}] Memory after predictions: {get_mem_gb():.3f} GB")

    # ========================================================================
    # CHECKPOINT SAVING
    # ========================================================================
    
    print("=" * 70)
    print(f"[Model {MODEL_ID}] Training complete, saving checkpoint...")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'model_{MODEL_ID}_session_{CURRENT_SESSION}.pkl')
    
    checkpoint = {
        'model_id': MODEL_ID,
        'weights': weights,
        'history': history,
        'session': CURRENT_SESSION,
        'timestamp': datetime.now().isoformat(),
        'total_samples': cumulative_samples,
        'total_time': time.time() - session_start_time,
        'test_predictions': test_predictions,
        'test_labels': y_test if CURRENT_SESSION == N_SESSIONS else None,
        'smote_applied': USE_SMOTE and original_fire_pct < SMOTE_TARGET_RATIO,
        'original_fire_pct': original_fire_pct,
        'balanced_fire_pct': y_session.mean() if USE_SMOTE else original_fire_pct,
        'config': config
    }
    
    with open(ckpt_path, 'wb') as f:
        pickle.dump(checkpoint, f, protocol=4)
        
    print(f"[Model {MODEL_ID}] Checkpoint saved -> {ckpt_path}")
    if CURRENT_SESSION == N_SESSIONS:
        print(f"[Model {MODEL_ID}] Test predictions cached in checkpoint")
    print(f"[Model {MODEL_ID}] Session {CURRENT_SESSION} complete.")
    print(f"[Model {MODEL_ID}] Final memory: {get_mem_gb():.3f} GB")
    print("=" * 70)
    
except Exception as e:
    print(f"[ERROR] Exception in training loop: {e}")
    traceback.print_exc()
    sys.exit(1)
