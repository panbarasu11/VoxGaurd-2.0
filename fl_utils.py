"""
fl_utils.py
-----------
Federated Learning Utilities for VoxGuard:
1. Per-Coefficient Normalization (Mean & Std per MFCC channel across frames)
2. Class-Balanced Data Partitioning across Edge Clients
3. Federated Averaging (FedAvg) aggregation algorithm
"""

import os
import numpy as np


NORM_STATS_PATH = "data/norm_stats.npz"


def compute_and_save_norm_stats(X, path=NORM_STATS_PATH):
    """Compute per-coefficient mean and std across dataset and save to disk."""
    mean_coeff = np.mean(X, axis=(0, 1), keepdims=True)  # shape (1, 1, 40)
    std_coeff = np.std(X, axis=(0, 1), keepdims=True) + 1e-8  # shape (1, 1, 40)
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, mean=mean_coeff, std=std_coeff)
    return mean_coeff, std_coeff


def load_norm_stats(path=NORM_STATS_PATH):
    """Load per-coefficient mean and std from disk."""
    if os.path.exists(path):
        data = np.load(path)
        return data["mean"], data["std"]
    return None, None


def normalize_features(X, mean_coeff=None, std_coeff=None):
    """Normalize MFCC features using per-coefficient mean and std."""
    if mean_coeff is None or std_coeff is None:
        mean_coeff = np.mean(X, axis=(0, 1), keepdims=True)
        std_coeff = np.std(X, axis=(0, 1), keepdims=True) + 1e-8
        
    return (X - mean_coeff) / std_coeff


def balance_dataset(X, y, seed=42):
    """Balance real (0) and fake (1) audio classes to eliminate label bias."""
    np.random.seed(seed)
    real_idx = np.where(y == 0)[0]
    fake_idx = np.where(y == 1)[0]
    
    min_count = min(len(real_idx), len(fake_idx))
    
    sel_real = np.random.choice(real_idx, size=min_count, replace=False)
    sel_fake = np.random.choice(fake_idx, size=min_count, replace=False)
    
    balanced_indices = np.concatenate([sel_real, sel_fake])
    np.random.shuffle(balanced_indices)
    
    return X[balanced_indices], y[balanced_indices]


def partition_data_iid(X, y, num_clients, seed=42):
    """Partition dataset into IID client subsets."""
    np.random.seed(seed)
    num_samples = len(X)
    shuffled_indices = np.random.permutation(num_samples)
    client_indices_list = np.array_split(shuffled_indices, num_clients)
    
    client_datasets = []
    for indices in client_indices_list:
        client_datasets.append((X[indices], y[indices]))
        
    return client_datasets


def partition_data_non_iid(X, y, num_clients, alpha=0.5, seed=42):
    """Partition dataset into Non-IID client subsets using Dirichlet distribution."""
    np.random.seed(seed)
    unique_classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]
    
    for k in unique_classes:
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_k_splits = np.split(idx_k, proportions)
        
        for i, split in enumerate(idx_k_splits):
            client_indices[i].extend(split)
            
    client_datasets = []
    for indices in client_indices:
        indices = np.array(indices)
        np.random.shuffle(indices)
        client_datasets.append((X[indices], y[indices]))
        
    return client_datasets


def federated_averaging(client_weights_list, client_sizes):
    """Perform Federated Averaging (FedAvg) aggregation."""
    total_samples = sum(client_sizes)
    num_layers = len(client_weights_list[0])
    aggregated_weights = [np.zeros_like(w) for w in client_weights_list[0]]
    
    for weights, n_k in zip(client_weights_list, client_sizes):
        weight_factor = n_k / total_samples
        for layer_idx in range(num_layers):
            aggregated_weights[layer_idx] += weight_factor * weights[layer_idx]
            
    return aggregated_weights


def display_client_distribution(client_datasets):
    """Print summary table of data distribution across edge clients."""
    print("\n" + "=" * 65)
    print("  FL Edge Client Data Partition Summary")
    print("=" * 65)
    print(f"{'Client ID':<12}{'Total Samples':<16}{'Real (0)':<12}{'Fake (1)':<12}{'Fake Ratio (%)':<15}")
    print("-" * 65)
    
    total_all = 0
    for idx, (X_c, y_c) in enumerate(client_datasets):
        n_total = len(y_c)
        n_real = int(np.sum(y_c == 0))
        n_fake = int(np.sum(y_c == 1))
        fake_ratio = (n_fake / n_total * 100) if n_total > 0 else 0.0
        total_all += n_total
        print(f"Client #{idx:<5}{n_total:<16}{n_real:<12}{n_fake:<12}{fake_ratio:<15.2f}%")
        
    print("-" * 65)
    print(f"Total across {len(client_datasets)} clients: {total_all} samples\n")
