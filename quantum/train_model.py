#!/usr/bin/env python3
"""
Simplified Quantum Model Training

This script trains a single quantum variational circuit for wildfire prediction.
Simplified version for demonstration - trains one model through all sessions.

Usage:
    python train_model.py [OPTIONS]
    
    Optional arguments:
    --data PATH      : Path to preprocessed data (default: ./preprocessed/model_1.pkl)
    --output DIR     : Checkpoint output directory (default: ./checkpoints)
    --model-id N     : Which model config to use (default: 1, range: 1-12)
    --samples N      : Samples per session (default: 5000)
    --iterations N   : Max iterations per session (default: 30)
    --sessions N     : Number of training sessions (default: 4)

Example:
    # Train with defaults (model 1, 4 sessions)
    python train_model.py
    
    # Train model 3 with custom settings
    python train_model.py --model-id 3 --samples 2000 --iterations 20
"""

import sys
import os
import time
import pickle
import argparse
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train single quantum VQC model')
    
    parser.add_argument('--data', type=str, default='./preprocessed/model_1.pkl',
                       help='Path to preprocessed data')
    parser.add_argument('--output', type=str, default='./checkpoints',
                       help='Checkpoint output directory')
    parser.add_argument('--model-id', type=int, default=1, choices=range(1, 13),
                       help='Model configuration (1-12)')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Samples per training session')
    parser.add_argument('--iterations', type=int, default=30,
                       help='Max iterations per session')
    parser.add_argument('--sessions', type=int, default=4,
                       help='Number of training sessions')
    parser.add_argument('--threads', type=int, default=4,
                       help='OpenMP thread count')
    
    return parser.parse_args()

args = parse_args()

# Set environment
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.makedirs(args.output, exist_ok=True)

print("=" * 70)
print("SIMPLIFIED QUANTUM MODEL TRAINING")
print("=" * 70)
print(f"Model ID: {args.model_id}")
print(f"Data: {args.data}")
print(f"Output: {args.output}")
print(f"Sessions: {args.sessions}")
print(f"Samples/session: {args.samples}")
print(f"Iterations/session: {args.iterations}")
print(f"Started: {datetime.now().isoformat()}")
print("=" * 70)
print()

# ============================================================================
# MEMORY UTILITY
# ============================================================================

try:
    import psutil
    def get_mem_gb():
        return psutil.Process().memory_info().rss / 1e9
except Exception:
    def get_mem_gb():
        return -1.0

# ============================================================================
# IMPORTS
# ============================================================================

try:
    import pennylane as qml
    from pennylane import numpy as np
    print("[OK] PennyLane imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import PennyLane: {e}")
    print("Install with: pip install pennylane pennylane-lightning")
    sys.exit(1)

# ============================================================================
# MODEL CONFIGURATIONS
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

config = MODEL_CONFIGS[args.model_id]
N_QUBITS = 6
N_PROCESSING_QUBITS = 6
TOTAL_QUBITS = N_QUBITS + N_PROCESSING_QUBITS
PATIENCE = 5

print(f"Model config: LR={config['learning_rate']}, Layers={config['n_layers']}, Init={config['init_scale']:.2f}")
print()

# ============================================================================
# DATA LOADING
# ============================================================================

try:
    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}")
        print("Run preprocessing first: python preprocess.py")
        sys.exit(1)
    
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train'].astype(np.float32)
    y_train = data['y_train'].astype(int)
    X_test = data['X_test'].astype(np.float32)
    y_test = data['y_test'].astype(int)
    
    print(f"[OK] Loaded data: Train={len(X_train)}, Test={len(X_test)}")
    print(f"     Class balance: {y_train.mean():.3%} fires")
    print(f"     Memory: {get_mem_gb():.3f} GB")
    print()
    
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    sys.exit(1)

# ============================================================================
# QUANTUM CIRCUIT
# ============================================================================

dev = qml.device('lightning.qubit', wires=TOTAL_QUBITS)

@qml.qnode(dev)
def circuit(weights, x):
    """Variational quantum circuit for binary classification."""
    # L2 normalize input
    x_norm = x / (np.linalg.norm(x) + 1e-8)
    
    # Amplitude embedding
    qml.AmplitudeEmbedding(x_norm, wires=range(N_QUBITS), normalize=True)
    
    # Parameterized layers
    for layer in range(config['n_layers']):
        for qubit in range(TOTAL_QUBITS):
            qml.RY(weights[layer, qubit, 0], wires=qubit)
            qml.RZ(weights[layer, qubit, 1], wires=qubit)
        
        for qubit in range(TOTAL_QUBITS - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
    
    return qml.expval(qml.PauliZ(0))

print(f"[OK] Circuit: {config['n_layers']} layers, {TOTAL_QUBITS} qubits")
print(f"     Parameters: {config['n_layers'] * TOTAL_QUBITS * 2}")
print()

# ============================================================================
# TRAINING LOOP
# ============================================================================

# Initialize weights
weights_shape = (config['n_layers'], TOTAL_QUBITS, 2)
weights = np.random.uniform(-config['init_scale'], config['init_scale'], weights_shape)

# Create optimizer
opt = qml.GradientDescentOptimizer(stepsize=config['learning_rate'])

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

overall_start = time.time()
all_history = []

# Train through all sessions
for session in range(1, args.sessions + 1):
    print(f"\n{'='*70}")
    print(f"SESSION {session}/{args.sessions}")
    print(f"{'='*70}\n")
    
    session_start = time.time()
    
    # Select data for this session
    start_idx = (session - 1) * args.samples
    end_idx = min(start_idx + args.samples, len(X_train))
    X_session = X_train[start_idx:end_idx]
    y_session = y_train[start_idx:end_idx]
    
    print(f"Samples: {len(X_session)} (indices {start_idx}-{end_idx})")
    print(f"Fire rate: {y_session.mean():.3%}")
    print()
    
    # Early stopping
    best_loss = float('inf')
    best_weights = weights.copy()
    patience_counter = 0
    
    # Training iterations
    for iteration in range(args.iterations):
        iter_start = time.time()
        
        # Forward pass
        preds = np.array([circuit(weights, x) for x in X_session])
        y_pred = (preds > 0).astype(int)
        
        # Class-weighted loss
        n_pos = y_session.sum()
        n_neg = len(y_session) - n_pos
        
        if n_pos > 0 and n_neg > 0:
            weight_for_0 = len(y_session) / (2 * n_neg)
            weight_for_1 = len(y_session) / (2 * n_pos)
            sample_weights = np.where(y_session == 1, weight_for_1, weight_for_0)
        else:
            sample_weights = np.ones(len(y_session))
        
        loss = np.mean(sample_weights * (preds - (2 * y_session - 1)) ** 2)
        acc = np.mean(y_pred == y_session)
        
        # Gradient descent
        def cost_fn(w):
            p = np.array([circuit(w, x) for x in X_session])
            return np.mean(sample_weights * (p - (2 * y_session - 1)) ** 2)
        
        weights = opt.step(cost_fn, weights)
        
        # Metrics
        iter_time = time.time() - iter_start
        
        print(f"  Iter {iteration+1:2d}/{args.iterations}: "
              f"Loss={loss:.6f}, Acc={acc:.4f}, Time={iter_time:.2f}s")
        
        all_history.append({
            'session': session,
            'iteration': iteration + 1,
            'loss': float(loss),
            'accuracy': float(acc),
            'time': iter_time
        })
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss
            best_weights = weights.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at iteration {iteration + 1}")
                weights = best_weights
                break
    
    session_time = time.time() - session_start
    print(f"\nSession {session} complete in {session_time:.1f}s ({session_time/60:.1f} min)")
    
    # Save checkpoint
    ckpt_path = os.path.join(args.output, f'model_{args.model_id}_session_{session}.pkl')
    checkpoint = {
        'model_id': args.model_id,
        'session': session,
        'weights': weights,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'session_time': session_time
    }
    
    with open(ckpt_path, 'wb') as f:
        pickle.dump(checkpoint, f, protocol=4)
    
    print(f"Saved checkpoint: {ckpt_path}")

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70 + "\n")

pred_start = time.time()
test_preds_raw = np.array([circuit(weights, x) for x in X_test])
test_preds = (test_preds_raw > 0).astype(int)
pred_time = time.time() - pred_start

# Calculate metrics
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, test_preds)
cm = confusion_matrix(y_test, test_preds)

if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (recall + specificity) / 2
else:
    recall = specificity = balanced_acc = 0

print(f"Test predictions generated in {pred_time:.1f}s")
print(f"Throughput: {len(X_test)/pred_time:.1f} samples/sec")
print()

print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Test Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Balanced Accuracy:      {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
print(f"Recall (Fire Detection):{recall:.4f} ({recall*100:.2f}%)")
print(f"Specificity:            {specificity:.4f} ({specificity*100:.2f}%)")
print()

# Save final results
final_path = os.path.join(args.output, f'model_{args.model_id}_final.pkl')
final_results = {
    'model_id': args.model_id,
    'config': config,
    'weights': weights,
    'test_predictions': test_preds,
    'test_labels': y_test,
    'metrics': {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'recall': recall,
        'specificity': specificity
    },
    'history': all_history,
    'total_time': time.time() - overall_start,
    'timestamp': datetime.now().isoformat()
}

with open(final_path, 'wb') as f:
    pickle.dump(final_results, f, protocol=4)

print(f"Final results saved: {final_path}")
print()

total_time = time.time() - overall_start
print("=" * 70)
print(f"TRAINING COMPLETE - Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print("=" * 70)
