import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    homogeneity_score,
)
from scipy.optimize import linear_sum_assignment
import torch
from sklearn.metrics.cluster import contingency_matrix

# Evaluation of the cluster properties for thesis

metadata_path = "metadata/task_embedding/roberta_small/metaworld-all.json"
#path_own_loss = 'Transformer_RNN/embedding_log/emb_own.pth'
path_own_loss = 'Transformer_RNN/embedding_log/emb.pth'
path_std_loss = 'Transformer_RNN/embedding_log/emb_std.pth'
path_Rnn = 'Transformer_RNN/embedding_log/rnn_emb.pth'

# Function to calculate accuracy using the Hungarian algorithm
def calculate_acc(y_true, y_pred):
    """
    Match predicted clusters to ground truth labels and calculate accuracy.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted cluster labels

    Returns:
    - accuracy: Accuracy of clustering
    """
    contingency_matrix = np.zeros((max(y_true) + 1, max(y_pred) + 1))
    for i, j in zip(y_true, y_pred):
        contingency_matrix[i, j] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return sum(contingency_matrix[row, col] for row, col in zip(row_ind, col_ind)) / len(y_true)

def calculate_acc_multi_assignments(y_true, y_pred):
    """
    Calculate Accuracy (ACC) with multiple assignments allowed per cluster.

    Parameters:
    - y_true: Ground truth labels.
    - y_pred: Predicted cluster labels.

    Returns:
    - accuracy: Accuracy considering multiple assignments.
    """
    # Build contingency matrix
    cont_matrix = contingency_matrix(y_true, y_pred)
    
    # Assign clusters to labels maximizing accuracy
    # Each cluster contributes its best-matching class count
    max_assignments = cont_matrix.max(axis=0)  # Max overlap for each predicted cluster
    accuracy = max_assignments.sum() / len(y_true)  # Normalize by total samples
    
    return accuracy

# Example dataset
# Replace these with your own dataset and labels
seed=1
np.random.seed(seed)
n_clusters = 10

def Rnn():
    datafile = torch.load(path_own_loss)
    data = datafile['state_emb']
    env_idx = datafile['task']
    unique_env_idx = np.unique(env_idx)
    value_to_rank = {value: rank-1 for rank, value in enumerate(unique_env_idx, start=1)}
    replaced_env_idx = np.vectorize(value_to_rank.get)(env_idx)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    y_pred = kmeans.fit_predict(data)

    # Calculate metrics
    acc_adjusted = calculate_acc_multi_assignments(replaced_env_idx, y_pred)
    silhouette = silhouette_score(data, y_pred)
    homogeneity = homogeneity_score(replaced_env_idx, y_pred)

    # Print results
    print(f"Adjusted Accuracy (ACC): {acc_adjusted:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Homogeneity Score: {homogeneity:.4f}")

def transformer(path):
    datafile = torch.load(path)
    data = datafile['tra_emb']
    env_idx = datafile['task']
    unique_env_idx = np.unique(env_idx)
    value_to_rank = {value: rank-1 for rank, value in enumerate(unique_env_idx, start=1)}
    replaced_env_idx = np.vectorize(value_to_rank.get)(env_idx)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    y_pred = kmeans.fit_predict(data)

    # Calculate metrics
    acc_adjusted = calculate_acc_multi_assignments(replaced_env_idx, y_pred)
    silhouette = silhouette_score(data, y_pred)
    homogeneity = homogeneity_score(replaced_env_idx, y_pred)

    # Print results
    print(f"Adjusted Accuracy (ACC): {acc_adjusted:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Homogeneity Score: {homogeneity:.4f}")

print(f"--- Evaluation own loss with {n_clusters} clusters ---")
transformer(path_own_loss)
# print(f"--- Evaluation stdandard loss with {n_clusters} clusters ---")
# transformer(path_std_loss)
# print(f"--- Evaluation Rnn with {n_clusters} clusters ---")
# Rnn()