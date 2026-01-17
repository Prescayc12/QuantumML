#!/usr/bin/env python3
"""
Parallel Quantum Training Orchestrator

Manages distributed training of 12 quantum models across multiple sessions.
Launches training workers in batches, monitors completion, and tracks progress.

This script is designed for advanced users running full ensemble training.
For simple single-model training, use train_model.py instead.

Usage:
    python orchestrator.py [OPTIONS]
    
    Optional arguments:
    --models N           : Number of models to train (default: 12)
    --sessions N         : Number of training sessions (default: 4)
    --batch-size N       : Models to train in parallel (default: 4)
    --data-dir PATH      : Preprocessed data directory (default: ./preprocessed)
    --checkpoint-dir PATH: Checkpoint output directory (default: ./checkpoints)
    --samples N          : Samples per session (default: 5000)
    --iterations N       : Iterations per session (default: 30)
    
Environment-specific notes:
    This orchestrator was originally designed for WSL2 + Windows Terminal.
    You may need to modify the launch commands for your specific environment.
    
    Current implementation launches workers using Python subprocess.
    For WSL2-specific Windows Terminal launching, see the commented section below.
"""

import subprocess
import time
import os
import sys
import argparse
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Parallel quantum training orchestrator')
    
    parser.add_argument('--models', type=int, default=12,
                       help='Number of models to train')
    parser.add_argument('--sessions', type=int, default=4,
                       help='Number of training sessions per model')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of models to train in parallel')
    parser.add_argument('--data-dir', type=str, default='./preprocessed',
                       help='Preprocessed data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint output directory')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Samples per training session')
    parser.add_argument('--iterations', type=int, default=30,
                       help='Max iterations per session')
    parser.add_argument('--worker-script', type=str, default='train_worker.py',
                       help='Training worker script name')
    
    return parser.parse_args()

args = parse_args()
WORK_DIR = os.getcwd()

# ============================================================================
# PREREQUISITE CHECKS
# ============================================================================

def check_preprocessed_data():
    """Verify presence of all preprocessed model shards."""
    if not os.path.exists(args.data_dir):
        return False

    missing = []
    for model_id in range(1, args.models + 1):
        pkl_path = os.path.join(args.data_dir, f'model_{model_id}.pkl')
        if not os.path.exists(pkl_path):
            missing.append(model_id)

    metadata_ok = os.path.exists(os.path.join(args.data_dir, 'metadata.pkl'))

    if missing or not metadata_ok:
        if missing:
            print(f"[ERROR] Missing preprocessed data for models: {missing}")
        if not metadata_ok:
            print(f"[ERROR] Missing metadata.pkl in {args.data_dir}")
        return False

    return True

def run_preprocessing():
    """Execute preprocessing script."""
    print("\n" + "="*70)
    print("RUNNING DATA PREPROCESSING")
    print("="*70 + "\n")

    if not os.path.exists('../data/Wildfire_Dataset.csv'):
        print("[ERROR] Dataset file missing: ../data/Wildfire_Dataset.csv")
        print("Download from: https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset")
        return False

    try:
        subprocess.run([
            'python', 'preprocess.py',
            '--output', args.data_dir,
            '--models', str(args.models)
        ], cwd=WORK_DIR, check=True)
        return True
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return False

# ============================================================================
# WORKER LAUNCHING
# ============================================================================

def launch_batch(model_ids, session):
    """
    Launch a batch of model training jobs.
    
    This implementation uses simple Python subprocess launching.
    For WSL2 + Windows Terminal, see the commented alternative below.
    """
    processes = []
    
    for model_id in model_ids:
        log_dir = os.path.join('logs', f'model_{model_id}')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'session_{session}.log')
        
        # Construct worker command
        cmd = [
            'python', args.worker_script,
            str(model_id), str(session),
            '--data', os.path.join(args.data_dir, f'model_{model_id}.pkl'),
            '--checkpoint-dir', args.checkpoint_dir,
            '--samples', str(args.samples),
            '--iterations', str(args.iterations)
        ]
        
        # Launch worker
        with open(log_path, 'w') as log_file:
            p = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=WORK_DIR
            )
            processes.append((model_id, p))
            print(f"  Launched Model {model_id} (PID: {p.pid}) -> {log_path}")
        
        time.sleep(1)  # Stagger launches slightly

    return processes

# ============================================================================
# ALTERNATIVE: WSL2 + WINDOWS TERMINAL LAUNCHING
# ============================================================================
"""
For WSL2 environments where you want to launch in separate Windows Terminal tabs:

def get_wsl_path(windows_path):
    path = windows_path.replace('\\', '/')
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path

def launch_batch_wsl(model_ids, session):
    processes = []
    wsl_work_dir = get_wsl_path(WORK_DIR)
    
    for model_id in model_ids:
        log_path = f"logs/model_{model_id}/session_{session}.log"
        
        wsl_cmd = (
            f"cd '{wsl_work_dir}' && "
            f"source venv/bin/activate && "
            f"mkdir -p logs/model_{model_id} && "
            f"export OMP_NUM_THREADS=4 && "
            f"python {args.worker_script} {model_id} {session} "
            f"--data {args.data_dir}/model_{model_id}.pkl "
            f"--checkpoint-dir {args.checkpoint_dir} "
            f"--samples {args.samples} "
            f"--iterations {args.iterations} "
            f"2>&1 | tee {log_path}"
        )
        
        cmd = [
            'wt.exe',
            '--title', f'Model {model_id} - Session {session}',
            'wsl', 'bash', '-c', wsl_cmd
        ]
        
        p = subprocess.Popen(cmd, shell=False)
        processes.append((model_id, p))
        time.sleep(2)
    
    return processes

# To use WSL2 launching, replace launch_batch() call with launch_batch_wsl()
"""

# ============================================================================
# MONITORING
# ============================================================================

def wait_for_batch(model_ids, session):
    """Poll checkpoint directory until all batch jobs complete."""
    completed = set()
    check_count = 0
    start_time = time.time()

    print(f"\n  Waiting for batch to complete...")
    
    while len(completed) < len(model_ids):
        check_count += 1
        newly_completed = []
        
        for model_id in model_ids:
            if model_id in completed:
                continue

            ckpt = os.path.join(args.checkpoint_dir, f'model_{model_id}_session_{session}.pkl')
            if os.path.exists(ckpt):
                completed.add(model_id)
                newly_completed.append(model_id)
        
        if newly_completed:
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.1f}s] Completed: {sorted(completed)} ({len(completed)}/{len(model_ids)})")
        
        if len(completed) < len(model_ids):
            time.sleep(5)
    
    total_time = time.time() - start_time
    print(f"  Batch complete in {total_time:.1f}s ({total_time/60:.1f} min)")

def verify_checkpoints(model_ids, session):
    """Ensure all checkpoints for the batch were produced."""
    all_ok = True
    missing = []
    
    for model_id in model_ids:
        ckpt = os.path.join(args.checkpoint_dir, f'model_{model_id}_session_{session}.pkl')
        if not os.path.exists(ckpt):
            all_ok = False
            missing.append(model_id)
    
    if not all_ok:
        print(f"  [WARNING] Missing checkpoints for models: {missing}")
    
    return all_ok

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    print("="*70)
    print("QUANTUM WILDFIRE PARALLEL TRAINING ORCHESTRATOR")
    print("="*70)
    print(f"Working directory: {WORK_DIR}")
    print(f"Configuration:")
    print(f"  Models: {args.models}")
    print(f"  Sessions: {args.sessions}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Samples/session: {args.samples}")
    print(f"  Iterations/session: {args.iterations}")
    print()

    # Check prerequisites
    if not check_preprocessed_data():
        resp = input("Preprocessed data missing. Run preprocessing now? (y/n): ").strip().lower()
        if resp == 'y':
            if not run_preprocessing() or not check_preprocessed_data():
                print("[ERROR] Preprocessing failed or incomplete")
                sys.exit(1)
        else:
            print("[ERROR] Cannot proceed without preprocessed data")
            sys.exit(1)

    # Create output directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("\n" + "="*70)
    print("STARTING DISTRIBUTED TRAINING")
    print("="*70)
    input("\nPress Enter to begin training...")

    overall_start = time.time()

    # Train through all sessions
    for session in range(1, args.sessions + 1):
        print(f"\n{'='*70}")
        print(f"SESSION {session}/{args.sessions}")
        print(f"{'='*70}")
        
        session_start = time.time()

        # Launch models in batches
        for batch_idx in range(0, args.models, args.batch_size):
            start = batch_idx + 1
            end = min(start + args.batch_size - 1, args.models)
            ids = list(range(start, end + 1))

            print(f"\nBatch: Models {start}-{end}")
            launch_batch(ids, session)
            wait_for_batch(ids, session)
            verify_checkpoints(ids, session)
            time.sleep(2)

        session_time = time.time() - session_start
        print(f"\nSession {session} complete: {session_time/60:.1f} min")

    total_time = time.time() - overall_start
    print("\n" + "="*70)
    print(f"TRAINING COMPLETE - Total time: {total_time/60:.1f} minutes")
    print("="*70)
    print("\nNext steps:")
    print(f"  1. Evaluate ensemble: python evaluate_ensemble.py --checkpoint-dir {args.checkpoint_dir}")
    print(f"  2. Optimize thresholds: python optimize_ensemble.py --checkpoint-dir {args.checkpoint_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
