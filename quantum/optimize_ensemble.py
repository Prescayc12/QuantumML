#!/usr/bin/env python3
"""
Quantum Ensemble Optimizer

Post-training optimization for quantum wildfire prediction ensemble.
Optimizes decision threshold and ensemble strategy without retraining.

This script:
1. Loads all trained quantum models (weights from final session)
2. Generates predictions on a common test set for all models
3. Tests 41 different probability thresholds (0.20 to 0.60)
4. Evaluates 4 ensemble strategies (simple average, F1-weighted, recall-weighted, meta-learner)
5. Selects optimal threshold and strategy based on balanced accuracy
6. Saves deployment-ready configuration

Note: This requires re-running quantum circuits to generate raw expectation values,
which is computationally expensive but necessary for proper threshold optimization.

Usage:
    python optimize_ensemble.py [OPTIONS]
    
    Optional arguments:
    --checkpoint-dir DIR  : Directory with trained model checkpoints (default: ./checkpoints)
    --data PATH          : Path to preprocessed data for test set (default: ./preprocessed/model_1.pkl)
    --output-dir DIR     : Output directory for config (default: ./config)
    --test-samples N     : Reduce test set to N samples for speed (default: 1000)
    --models N           : Number of models in ensemble (default: 12)
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import time
import argparse
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Optimize quantum ensemble thresholds and strategies')
    
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory with trained model checkpoints')
    parser.add_argument('--data', type=str, default='./preprocessed/model_1.pkl',
                       help='Path to preprocessed data (for test set)')
    parser.add_argument('--output-dir', type=str, default='./config',
                       help='Output directory for optimal configuration')
    parser.add_argument('--test-samples', type=int, default=1000,
                       help='Reduce test set to N samples for faster optimization')
    parser.add_argument('--models', type=int, default=12,
                       help='Number of models in ensemble')
    parser.add_argument('--sessions', type=int, default=4,
                       help='Number of training sessions (uses final session)')
    
    return parser.parse_args()

args = parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

print("=" * 70)
print("QUANTUM ENSEMBLE OPTIMIZER")
print("=" * 70)
print(f"Checkpoint dir: {args.checkpoint_dir}")
print(f"Data source: {args.data}")
print(f"Output dir: {args.output_dir}")
print(f"Test samples: {args.test_samples}")
print(f"Models: {args.models}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

N_QUBITS = 6
N_PROCESSING_QUBITS = 6
TOTAL_QUBITS = N_QUBITS + N_PROCESSING_QUBITS

# Model configurations (must match training)
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

# ============================================================================
# PENNYLANE IMPORT
# ============================================================================

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    print("[OK] PennyLane imported successfully")
except ImportError:
    print("[ERROR] PennyLane not found. Install with: pip install pennylane pennylane-lightning")
    sys.exit(1)

# ============================================================================
# QUANTUM CIRCUIT DEFINITION
# ============================================================================

def create_quantum_circuit(n_qubits, n_proc, n_layers):
    """Create quantum circuit matching the training architecture."""
    total = n_qubits + n_proc
    dev = qml.device('lightning.qubit', wires=total)

    @qml.qnode(dev)
    def circuit(weights, x):
        # L2 normalize input
        x_norm = x / (pnp.linalg.norm(x) + 1e-8)
        
        # Amplitude embedding
        qml.AmplitudeEmbedding(x_norm, wires=range(n_qubits), normalize=True)
        
        # Parameterized layers
        for l in range(n_layers):
            for i in range(total):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            
            for i in range(total - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return qml.expval(qml.PauliZ(0))

    return circuit

# ============================================================================
# MODEL AND DATA LOADING
# ============================================================================

def load_models_and_data():
    """Load trained model weights and common test dataset."""
    print("Loading trained models and test data...")
    
    models = []
    
    # Load each model's final checkpoint
    for model_id in range(1, args.models + 1):
        ckpt_path = os.path.join(args.checkpoint_dir, f'model_{model_id}_session_{args.sessions}.pkl')
        
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Model {model_id} checkpoint not found: {ckpt_path}")
            continue
        
        with open(ckpt_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        config = MODEL_CONFIGS[model_id]
        
        # Create circuit for this model
        circuit = create_quantum_circuit(N_QUBITS, N_PROCESSING_QUBITS, config['n_layers'])
        
        # Get validation metrics from checkpoint (if available)
        val_f1 = checkpoint.get('val_f1', 0.1)
        val_recall = checkpoint.get('val_recall', 0.5)
        
        models.append({
            'model_id': model_id,
            'weights': checkpoint['weights'],
            'circuit': circuit,
            'config': config,
            'val_f1': val_f1,
            'val_recall': val_recall
        })
        
        print(f"  [OK] Model {model_id} loaded (layers={config['n_layers']})")
    
    print(f"\nLoaded {len(models)}/{args.models} models")
    
    # Load common test data
    print(f"\nLoading test dataset from {args.data}...")
    
    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}")
        sys.exit(1)
    
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    
    X_test = data['X_test'].astype(np.float32)
    y_test = data['y_test'].astype(int)
    
    print(f"Original test set size: {len(X_test)} samples")
    print(f"Class balance: {y_test.mean():.3%} fires")
    
    # Reduce to subset for faster optimization
    if len(X_test) > args.test_samples:
        print(f"\nReducing to {args.test_samples} samples for computational efficiency...")
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, 
            train_size=args.test_samples, 
            stratify=y_test,
            random_state=42
        )
        print(f"Reduced test set size: {len(X_test)} samples")
        print(f"Maintained class balance: {y_test.mean():.3%} fires")
    
    return models, X_test, y_test

# ============================================================================
# PREDICTION GENERATION
# ============================================================================

def generate_ensemble_predictions(models, X_test, y_test):
    """Generate predictions from all models on common test set."""
    print("\nGenerating predictions from all models...")
    print("(This will take a while - re-running quantum circuits)")
    
    all_predictions = []
    model_info = []
    
    for model in models:
        model_id = model['model_id']
        print(f"\n  Model {model_id}:")
        
        start_time = time.time()
        
        # Generate predictions (expensive quantum circuit evaluation)
        expectation_values = []
        batch_size = 50
        
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:min(i+batch_size, len(X_test))]
            batch_preds = [model['circuit'](model['weights'], x) for x in batch]
            expectation_values.extend(batch_preds)
            
            if (i // batch_size) % 5 == 0:
                progress = min(i + batch_size, len(X_test)) / len(X_test)
                print(f"    Progress: {progress*100:.1f}%")
        
        expectation_values = np.array(expectation_values)
        elapsed = time.time() - start_time
        
        print(f"    Complete in {elapsed:.1f}s ({len(X_test)/elapsed:.1f} samples/sec)")
        print(f"    Expectation range: [{expectation_values.min():.3f}, {expectation_values.max():.3f}]")
        
        all_predictions.append(expectation_values)
        model_info.append({
            'model_num': model_id,
            'val_f1': model['val_f1'],
            'val_recall': model['val_recall']
        })
    
    predictions = np.array(all_predictions)
    print(f"\nPrediction generation complete")
    print(f"Shape: {predictions.shape} (models × samples)")
    
    return predictions, model_info

# ============================================================================
# PROBABILITY CONVERSION
# ============================================================================

def expectation_to_probability(expectations):
    """Convert expectation values [-1, 1] to probabilities [0, 1]."""
    return (expectations + 1) / 2

# ============================================================================
# MODEL FILTERING
# ============================================================================

def filter_working_models(individual_probs, model_info, targets):
    """Remove models that failed to train properly."""
    working_indices = []
    working_info = []
    
    for i, info in enumerate(model_info):
        probs = individual_probs[i]
        # Model is valid if it makes diverse predictions
        unique_preds = len(np.unique((probs >= 0.5).astype(int)))
        
        if unique_preds > 1:
            working_indices.append(i)
            working_info.append(info)
    
    working_preds = individual_probs[working_indices]
    
    print(f"\nActive models: {len(working_indices)}/{len(model_info)}")
    
    return working_preds, working_info

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def find_optimal_thresholds(probs, targets):
    """Find threshold for highest accuracy and highest balanced accuracy."""
    thresholds = np.arange(0.20, 0.61, 0.01)
    results = []
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (recall + specificity) / 2
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(targets, preds, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'recall': recall,
            'specificity': specificity,
            'balanced_accuracy': balanced_acc,
            'accuracy': accuracy,
            'precision': precision,
            'f1': f1,
            'false_alarms': fp
        })
    
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    best_balanced = max(results, key=lambda x: x['balanced_accuracy'])
    
    return best_accuracy, best_balanced, results

# ============================================================================
# ENSEMBLE STRATEGY EVALUATION
# ============================================================================

def evaluate_ensemble_strategies(working_preds, working_info, targets, threshold):
    """Test different ensemble combination strategies."""
    strategies = {}
    
    # Strategy 1: Simple average
    avg_probs = working_preds.mean(axis=0)
    avg_preds = (avg_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, avg_preds).ravel()
    strategies['simple_average'] = {
        'probs': avg_probs,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
        'f1': f1_score(targets, avg_preds, zero_division=0)
    }
    
    # Strategy 2: F1-weighted
    f1_scores = np.array([info['val_f1'] for info in working_info])
    if f1_scores.sum() > 0:
        f1_weights = f1_scores / f1_scores.sum()
        weighted_probs = (working_preds.T @ f1_weights)
        weighted_preds = (weighted_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, weighted_preds).ravel()
        strategies['f1_weighted'] = {
            'probs': weighted_probs,
            'weights': f1_weights,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
            'f1': f1_score(targets, weighted_preds, zero_division=0)
        }
    else:
        strategies['f1_weighted'] = strategies['simple_average']
    
    # Strategy 3: Recall-weighted
    recall_scores = np.array([info['val_recall'] for info in working_info])
    if recall_scores.sum() > 0:
        recall_weights = recall_scores / recall_scores.sum()
        recall_weighted_probs = (working_preds.T @ recall_weights)
        recall_weighted_preds = (recall_weighted_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, recall_weighted_preds).ravel()
        strategies['recall_weighted'] = {
            'probs': recall_weighted_probs,
            'weights': recall_weights,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
            'f1': f1_score(targets, recall_weighted_preds, zero_division=0)
        }
    else:
        strategies['recall_weighted'] = strategies['simple_average']
    
    # Strategy 4: Meta-learner (Random Forest stacking)
    split_point = len(targets) // 2
    meta_train_preds = working_preds[:, :split_point].T
    meta_train_targets = targets[:split_point]
    meta_test_preds = working_preds[:, split_point:].T
    meta_test_targets = targets[split_point:]
    
    rf_meta = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf_meta.fit(meta_train_preds, meta_train_targets)
    rf_probs = rf_meta.predict_proba(meta_test_preds)[:, 1]
    rf_preds = (rf_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(meta_test_targets, rf_preds).ravel()
    strategies['meta_learner'] = {
        'probs': rf_probs,
        'model': rf_meta,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
        'f1': f1_score(meta_test_targets, rf_preds, zero_division=0),
        'note': 'Evaluated on 50% of test set'
    }
    
    # Find best strategy
    comparable_strategies = {k: v for k, v in strategies.items() if k != 'meta_learner'}
    best_name = max(comparable_strategies.keys(), key=lambda k: comparable_strategies[k]['balanced_accuracy'])
    
    return strategies, best_name

# ============================================================================
# CONFIGURATION SAVING
# ============================================================================

def save_optimal_config(threshold, strategy_name, strategy, working_info):
    """Save optimal configuration for deployment."""
    config = {
        'threshold': threshold,
        'ensemble_strategy': strategy_name,
        'weights': strategy.get('weights', None),
        'working_model_indices': [info['model_num'] for info in working_info],
        'performance': {
            'balanced_accuracy': strategy['balanced_accuracy'],
            'f1': strategy['f1'],
            'recall': strategy['recall'],
            'specificity': strategy['specificity']
        }
    }
    
    config_path = os.path.join(args.output_dir, 'optimal_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"\n[OK] Saved optimal configuration to {config_path}")

def create_threshold_plot(threshold_results, optimal_threshold):
    """Visualize threshold optimization results."""
    df = pd.DataFrame(threshold_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Recall vs Specificity
    axes[0].plot(df['threshold'], df['recall'], 'o-', label='Recall', linewidth=2, markersize=4)
    axes[0].plot(df['threshold'], df['specificity'], 's-', label='Specificity', linewidth=2, markersize=4)
    axes[0].axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='75% Target')
    axes[0].axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.5, label=f'Optimal: {optimal_threshold:.2f}')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Recall vs Specificity Trade-off')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Balanced Accuracy
    axes[1].plot(df['threshold'], df['balanced_accuracy'], 'o-', color='purple', linewidth=2, markersize=4)
    axes[1].axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.5, label=f'Optimal: {optimal_threshold:.2f}')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Balanced Accuracy')
    axes[1].set_title('Balanced Accuracy vs Threshold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, 'threshold_optimization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved threshold plot to {plot_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load models and data
    models, X_test, y_test = load_models_and_data()
    
    if len(models) == 0:
        print("[ERROR] No models loaded. Cannot proceed.")
        return
    
    # Generate ensemble predictions
    expectation_values, model_info = generate_ensemble_predictions(models, X_test, y_test)
    
    # Convert to probabilities
    print("\nConverting expectation values to probabilities...")
    individual_probs = expectation_to_probability(expectation_values)
    
    # Filter working models
    working_preds, working_info = filter_working_models(individual_probs, model_info, y_test)
    
    if len(working_info) == 0:
        print("[ERROR] No working models found. Cannot optimize ensemble.")
        return
    
    # Find optimal threshold
    print("\nOptimizing threshold...")
    avg_probs = working_preds.mean(axis=0)
    optimal_accuracy, optimal_balanced, all_results = find_optimal_thresholds(avg_probs, y_test)
    
    # Display results
    print(f"\nHighest Overall Accuracy:")
    print(f"  Threshold: {optimal_accuracy['threshold']:.2f}")
    print(f"  Overall Accuracy: {optimal_accuracy['accuracy']:.4f} ({optimal_accuracy['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy: {optimal_accuracy['balanced_accuracy']:.4f} ({optimal_accuracy['balanced_accuracy']*100:.2f}%)")
    print(f"  F1 Score: {optimal_accuracy['f1']:.4f}")
    print(f"  Recall: {optimal_accuracy['recall']:.4f}")
    print(f"  Specificity: {optimal_accuracy['specificity']:.4f}")
    print(f"  False Alarms: {optimal_accuracy['false_alarms']:,}")
    
    print(f"\nHighest Balanced Accuracy:")
    print(f"  Threshold: {optimal_balanced['threshold']:.2f}")
    print(f"  Balanced Accuracy: {optimal_balanced['balanced_accuracy']:.4f} ({optimal_balanced['balanced_accuracy']*100:.2f}%)")
    print(f"  Overall Accuracy: {optimal_balanced['accuracy']:.4f} ({optimal_balanced['accuracy']*100:.2f}%)")
    print(f"  F1 Score: {optimal_balanced['f1']:.4f}")
    print(f"  Recall: {optimal_balanced['recall']:.4f}")
    print(f"  Specificity: {optimal_balanced['specificity']:.4f}")
    print(f"  False Alarms: {optimal_balanced['false_alarms']:,}")
    
    # Test ensemble strategies
    print("\nEvaluating ensemble strategies...")
    strategies, best_strategy_name = evaluate_ensemble_strategies(
        working_preds, working_info, y_test, optimal_balanced['threshold']
    )
    
    # Display strategy comparison
    print("\nStrategy comparison:")
    for name, strategy in strategies.items():
        note = f" ({strategy['note']})" if 'note' in strategy else ""
        print(f"  {name:20s}: BA={strategy['balanced_accuracy']:.4f} "
              f"F1={strategy['f1']:.4f} Recall={strategy['recall']:.4f} Spec={strategy['specificity']:.4f}{note}")
    
    print(f"\nBest strategy: {best_strategy_name}")
    
    # Save optimal configuration
    save_optimal_config(
        optimal_balanced['threshold'],
        best_strategy_name,
        strategies[best_strategy_name],
        working_info
    )
    
    # Create visualization
    create_threshold_plot(all_results, optimal_balanced['threshold'])
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\nSaved files:")
    print(f"  {args.output_dir}/optimal_config.pkl")
    print(f"  {args.output_dir}/threshold_optimization.png")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Optimization cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
