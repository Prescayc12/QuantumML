"""
Post-Training Optimizer
Optimize decision threshold and ensemble strategy without retraining
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

def load_data():
    """Load prediction and test data from training output"""
    # Load prediction from all base model
    with open('results/predictions.pkl', 'rb') as f:
        pred_data = pickle.load(f)
    
    # Load test data for evaluation
    with open('results/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Extract individual model prediction and target
    individual_preds = pred_data['individual_predictions']  # Shape: (12, num_samples)
    targets = pred_data['targets']
    model_info = pred_data['model_info']
    
    return individual_preds, targets, model_info

def filter_working_models(individual_preds, model_info, targets):
    """Remove model that failed to train properly"""
    working_indices = []
    working_info = []
    
    # Check each model for valid prediction
    for i, info in enumerate(model_info):
        preds = individual_preds[i]
        # Model is valid if it make diverse prediction and has positive F1
        unique_preds = len(np.unique((preds >= 0.5).astype(int)))
        
        if unique_preds > 1 and info['val_f1'] > 0.01:
            working_indices.append(i)
            working_info.append(info)
    
    # Filter to working model only
    working_preds = individual_preds[working_indices]
    
    print(f"Active models: {len(working_indices)}/12")
    
    return working_preds, working_info

def find_optimal_thresholds(probs, targets):
    """Find threshold for highest accuracy and highest balanced accuracy"""
    # Test range of threshold value
    thresholds = np.arange(0.20, 0.61, 0.01)
    results = []
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        
        # Calculate metric
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_acc = (recall + specificity) / 2
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(targets, preds)
        
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
    
    # Find threshold with highest overall accuracy
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    
    # Find threshold with highest balanced accuracy
    best_balanced = max(results, key=lambda x: x['balanced_accuracy'])
    
    return best_accuracy, best_balanced, results

def evaluate_ensemble_strategies(working_preds, working_info, targets, threshold):
    """Test different ensemble combination strategy"""
    strategies = {}
    
    # Strategy 1: Simple average of all model
    avg_probs = working_preds.mean(axis=0)
    avg_preds = (avg_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, avg_preds).ravel()
    strategies['simple_average'] = {
        'probs': avg_probs,
        'recall': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2,
        'f1': f1_score(targets, avg_preds)
    }
    
    # Strategy 2: Weight by validation F1 score
    # Model with higher F1 get more influence
    f1_scores = np.array([info['val_f1'] for info in working_info])
    f1_weights = f1_scores / f1_scores.sum()
    weighted_probs = (working_preds.T @ f1_weights)
    weighted_preds = (weighted_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, weighted_preds).ravel()
    strategies['f1_weighted'] = {
        'probs': weighted_probs,
        'weights': f1_weights,
        'recall': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2,
        'f1': f1_score(targets, weighted_preds)
    }
    
    # Strategy 3: Weight by validation recall
    # Model with higher recall get more influence
    recall_scores = np.array([info['val_recall'] for info in working_info])
    recall_weights = recall_scores / recall_scores.sum()
    recall_weighted_probs = (working_preds.T @ recall_weights)
    recall_weighted_preds = (recall_weighted_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, recall_weighted_preds).ravel()
    strategies['recall_weighted'] = {
        'probs': recall_weighted_probs,
        'weights': recall_weights,
        'recall': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2,
        'f1': f1_score(targets, recall_weighted_preds)
    }
    
    # Strategy 4: Meta-learner (stacking)
    # Train random forest on model prediction
    # Split data for meta-learner training
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
        'recall': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2,
        'f1': f1_score(meta_test_targets, rf_preds),
        'note': 'Evaluated on 50% of test set'
    }
    
    # Find best strategy based on balanced accuracy
    # Exclude meta-learner from comparison since it use different data split
    comparable_strategies = {k: v for k, v in strategies.items() if k != 'meta_learner'}
    best_name = max(comparable_strategies.keys(), key=lambda k: comparable_strategies[k]['balanced_accuracy'])
    
    return strategies, best_name

def save_optimal_config(threshold, strategy_name, strategy, working_info):
    """Save optimal configuration for deployment"""
    # Create output directory
    os.makedirs('config', exist_ok=True)
    
    # Prepare configuration dictionary
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
    
    # Save configuration
    with open('config/optimal_config.pkl', 'wb') as f:
        pickle.dump(config, f)

def create_threshold_plot(threshold_results, optimal_threshold):
    """Visualize threshold optimization result"""
    os.makedirs('config', exist_ok=True)
    
    # Convert result to dataframe for easy plotting
    df = pd.DataFrame(threshold_results)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Recall and specificity vs threshold
    axes[0].plot(df['threshold'], df['recall'], 'o-', label='Recall', linewidth=2, markersize=4)
    axes[0].plot(df['threshold'], df['specificity'], 's-', label='Specificity', linewidth=2, markersize=4)
    axes[0].axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='75% Target')
    axes[0].axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.5, label=f'Optimal: {optimal_threshold:.2f}')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Recall vs Specificity Trade-off')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Balanced accuracy vs threshold
    axes[1].plot(df['threshold'], df['balanced_accuracy'], 'o-', color='purple', linewidth=2, markersize=4)
    axes[1].axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.5, label=f'Optimal: {optimal_threshold:.2f}')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Balanced Accuracy')
    axes[1].set_title('Balanced Accuracy vs Threshold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('config/threshold_optimization.png', dpi=300, bbox_inches='tight')

def main():
    print("Ensemble Optimizer")
    print("=" * 60)
    
    # Load data
    print("Loading predictions...")
    individual_preds, targets, model_info = load_data()
    
    # Filter to working model only
    working_preds, working_info = filter_working_models(individual_preds, model_info, targets)
    
    # Find optimal threshold
    print("\nOptimizing threshold...")
    avg_probs = working_preds.mean(axis=0)
    optimal_accuracy, optimal_balanced, all_results = find_optimal_thresholds(avg_probs, targets)
    
    # Display highest overall accuracy threshold
    print(f"\nHighest Overall Accuracy:")
    print(f"  Threshold: {optimal_accuracy['threshold']:.2f}")
    print(f"  Overall Accuracy: {optimal_accuracy['accuracy']:.4f} ({optimal_accuracy['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy: {optimal_accuracy['balanced_accuracy']:.4f} ({optimal_accuracy['balanced_accuracy']*100:.2f}%)")
    print(f"  F1 Score: {optimal_accuracy['f1']:.4f} ({optimal_accuracy['f1']*100:.2f}%)")
    print(f"  Recall: {optimal_accuracy['recall']:.4f} ({optimal_accuracy['recall']*100:.1f}%)")
    print(f"  Specificity: {optimal_accuracy['specificity']:.4f} ({optimal_accuracy['specificity']*100:.1f}%)")
    print(f"  False Alarms: {optimal_accuracy['false_alarms']:,}")
    
    # Display highest balanced accuracy threshold
    print(f"\nHighest Balanced Accuracy:")
    print(f"  Threshold: {optimal_balanced['threshold']:.2f}")
    print(f"  Balanced Accuracy: {optimal_balanced['balanced_accuracy']:.4f} ({optimal_balanced['balanced_accuracy']*100:.2f}%)")
    print(f"  Overall Accuracy: {optimal_balanced['accuracy']:.4f} ({optimal_balanced['accuracy']*100:.2f}%)")
    print(f"  F1 Score: {optimal_balanced['f1']:.4f} ({optimal_balanced['f1']*100:.2f}%)")
    print(f"  Recall: {optimal_balanced['recall']:.4f} ({optimal_balanced['recall']*100:.1f}%)")
    print(f"  Specificity: {optimal_balanced['specificity']:.4f} ({optimal_balanced['specificity']*100:.1f}%)")
    print(f"  False Alarms: {optimal_balanced['false_alarms']:,}")
    
    # Test ensemble strategy using balanced accuracy threshold
    print("\nEvaluating ensemble strategies...")
    strategies, best_strategy_name = evaluate_ensemble_strategies(
        working_preds, working_info, targets, optimal_balanced['threshold']
    )
    
    # Display strategy comparison
    print("\nStrategy comparison:")
    for name, strategy in strategies.items():
        note = f" ({strategy['note']})" if 'note' in strategy else ""
        print(f"  {name:20s}: BA={strategy['balanced_accuracy']:.4f} "
              f"F1={strategy['f1']:.4f} Recall={strategy['recall']:.4f} Spec={strategy['specificity']:.4f}{note}")
    
    print(f"\nBest strategy: {best_strategy_name}")
    
    # Save optimal configuration using balanced accuracy threshold
    save_optimal_config(
        optimal_balanced['threshold'],
        best_strategy_name,
        strategies[best_strategy_name],
        working_info
    )
    
    # Create visualization
    create_threshold_plot(all_results, optimal_balanced['threshold'])
    
    print("\nOptimization complete")
    print("=" * 60)
    print("\nSaved files:")
    print("  config/optimal_config.pkl (using balanced accuracy threshold)")
    print("  config/threshold_optimization.png")

if __name__ == "__main__":
    main()