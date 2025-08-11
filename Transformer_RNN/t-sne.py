import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import os

metadata_path = "metadata/task_embedding/roberta_small/metaworld-all.json"
#path = 'Transformer_RNN/embedding_log/emb_own.pth'
#path = 'Transformer_RNN/embedding_log/emb_std.pth'
path = 'Transformer_RNN/embedding_log/emb.pth'
cluster_path = 'Transformer_RNN/bnpy_save/data/latent_samples_end.npz'
rnn_path = 'Transformer_RNN/embedding_log/rnn_emb.pth'
safe_arm = False

def multiple_samples():
    # Generate or load your data as a numpy array
    datafile = torch.load(path)
    state_embedding = datafile['state_embedding']
    task_obs = datafile['task'].flatten()
    tra_num = datafile['tra_num'].flatten()
    rewards = datafile['rewards'].flatten()
    rewards = rewards / np.max(rewards)
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(task_obs))))

    # Create mapping ids to names
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())

    # Initialize t-SNE object
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(state_embedding)

    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(task_obs))))

    # Plot the result
    plt.figure(figsize=(16, 12))

    for i, label in enumerate(np.unique(task_obs)):
        mask = (task_obs == label)
        plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], label=task_names[label], alpha=0.5)

    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Create legend
    legend_dict = {label: color for label, color in zip(task_names, colors)}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
    plt.legend(handles=handles, title='Class')

    plt.savefig('Transformer_RNN/graphics/tSNE_plot_state_emb.png', bbox_inches='tight')

    plt.show()

    # trajacetory pos plot
    plt.figure(figsize=(16, 12))

    colors = plt.cm.coolwarm(tra_num) #plt.cm.tab10(np.linspace(0, 1, len(np.unique(tra_num))))

    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors)

    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='trajacetory pos')

    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.savefig('Transformer_RNN/graphics/tSNE_state_emb.png', bbox_inches='tight')

    plt.show()

def single_samples():
    # Generate or load your data as a numpy array
    datafile = torch.load(path)
    #data = datafile['state_emb']
    data = datafile['tra_emb']
    task_obs = datafile['task'].flatten()
    rewards = datafile['rewards'].flatten()
    task_arm = datafile['task_arm'].flatten()
    rewards = rewards / np.max(rewards)
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(task_obs))))

    # Create mapping ids to names
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())

    # Initialize t-SNE object
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(data)

    # Plot the result
    plt.figure(figsize=(16, 12))

    for i, label in enumerate(np.unique(task_obs)):
        mask = (task_obs == label)
        plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], label=task_names[label], alpha=0.5)

    plt.title('t-SNE Cluster Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)

    # Create legend
    legend_dict = {task_names[np.unique(task_obs)[i]]: colors[i] for i in range(len(np.unique(task_obs)))}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
    plt.legend(handles=handles, title='Task', fontsize=13, title_fontsize=14)

    plt.savefig('Transformer_RNN/graphics/tSNE_plot.png', bbox_inches='tight')

    plt.show()

    # reward plot
    plt.figure(figsize=(16, 12))

    colors = plt.cm.coolwarm(rewards)

    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors)

    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='Reward')

    plt.title('Reward Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)

    plt.savefig('Transformer_RNN/graphics/tSNE_reward_plot.png', bbox_inches='tight')

    plt.show()

    # arm config plot
    if safe_arm:
        plt.figure(figsize=(16, 12))

        colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(task_arm))))

        for i in range(len(np.unique(task_arm))):
            mask = (task_arm == i)
            plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], alpha=0.5)

        plt.title('Arm Config Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        # Create legend
        legend_dict = {label: color for label, color in zip(np.unique(task_arm), colors)}
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
        plt.legend(handles=handles, title='Class')

        plt.savefig('Transformer_RNN/graphics/tSNE_arm_config_plot.png', bbox_inches='tight')

        plt.show()

def plot_trajectory_emb():
    # Generate or load your data as a numpy array
    datafile = torch.load(path)
    #data = datafile['state_emb']
    data = datafile['tra_emb']
    task_obs = datafile['task'].flatten()
    task_arm = datafile['task_arm'].flatten()
    rewards = datafile['rewards'].flatten()
    timesteps = datafile['timesteps'].flatten()
    timesteps = timesteps / np.max(timesteps)
    rewards = rewards / np.max(rewards)
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(task_obs))))

    # Create mapping ids to names
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())

    # Initialize t-SNE object
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(data)

    task_ids = np.unique(task_obs)
    arm_ids = np.unique(task_arm)
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_ids)))
    markers = ['o', '^', 's', 'D', 'P', 'X', '*', 'v']  
    assert len(arm_ids) <= len(markers), "please extend the markers list."

    # Plot the result
    plt.figure(figsize=(16, 12))

    task_handles = {}
    for ti, t in enumerate(task_ids):
        color = colors[ti]
        for ai, a in enumerate(arm_ids):
            m = markers[ai]
            mask = (task_obs == t) & (task_arm == a)
            if not np.any(mask):
                continue
            plt.scatter(
                data_tsne[mask, 0], data_tsne[mask, 1],
                color=[color], marker=m, alpha=0.6, s=18, linewidths=0
            )
        task_handles[t] = plt.Line2D([0], [0], marker='o', color='w',
                                     label=task_names[t],
                                     markerfacecolor=colors[ti], markersize=10)

    plt.title('t-SNE Cluster Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)

    leg_tasks = plt.legend(handles=list(task_handles.values()),
                           title='Task', fontsize=13, title_fontsize=14,
                           loc='lower right', frameon=True)
    plt.gca().add_artist(leg_tasks)

    arm_handles = [plt.Line2D([0],[0], marker=markers[i], color='k',
                              label=f'arm-{int(a)}', linestyle='',
                              markerfacecolor='none', markersize=10)
                   for i, a in enumerate(arm_ids)]
    plt.legend(handles=arm_handles, title='Robot', fontsize=13, title_fontsize=14,
               loc='lower center', frameon=True)

    os.makedirs('Transformer_RNN/graphics', exist_ok=True)
    plt.savefig('Transformer_RNN/graphics/tSNE_plot_tra.svg', format='svg', bbox_inches='tight')
    plt.show()


    # reward plot
    plt.figure(figsize=(16, 12))

    colors = plt.cm.coolwarm(rewards)

    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors)

    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='Reward')

    plt.title('Reward Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)

    plt.savefig('Transformer_RNN/graphics/tSNE_reward_plot_tra.svg', format='svg', bbox_inches='tight')

    plt.show()

    # timesteps plot
    plt.figure(figsize=(16, 12))

    colors = plt.cm.coolwarm(timesteps)

    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors)

    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='Timesteps')

    plt.title('Timesteps Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)

    plt.savefig('Transformer_RNN/graphics/tSNE_timesteps_plot_tra.svg', format='svg', bbox_inches='tight')

    plt.show()


def analysis():
    datafile = np.load(cluster_path)
    data = datafile['z']
    env_idx = datafile['env_idx']
    best = None
    score = -1

    for i in range(3):
        print(f"--- {i} ---")
        kmeans = KMeans(n_clusters=10, random_state=i)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(data, cluster_labels)
        print(f"Silhouette Score for {10} clusters: {silhouette_avg:.3f}")
        
        # Calculate Adjusted Rand Index (ARI) if true labels are provided
        ari = adjusted_rand_score(env_idx, cluster_labels)
        print(f"Adjusted Rand Index (ARI) for {10} clusters: {ari:.3f}")

        if silhouette_avg > score:
            best = cluster_labels

    cluster_labels = best

    unique_vals = np.unique(env_idx)
    sorted_unique_vals = np.sort(unique_vals)
    
    # Create a dictionary mapping each unique value to its rank
    rank_dict = {val: rank+1 for rank, val in enumerate(sorted_unique_vals)}
    
    # Use np.vectorize to replace each element with its rank using the dictionary
    env_idx = np.vectorize(rank_dict.get)(env_idx)

    reduced_data = TSNE(n_components=2).fit_transform(data)
    # reduced_data = TSNE(n_components=2, random_state=0).fit_transform(data)

    for e in np.unique(env_idx):
        mean = np.mean(data[env_idx==e], axis=0)
        variance = np.var(data[env_idx==e], axis=0)
        std_dev = np.std(data[env_idx==e], axis=0)
    
        print(f"Descriptive Statistics for {e}:")
        print("Mean:\n", mean)
        print("Variance:\n", variance)
        print("Standard Deviation:\n", std_dev)

    print(f"--- Clusters ---")
    for e in np.unique(cluster_labels):
        mean = np.mean(data[cluster_labels==e], axis=0)
        variance = np.var(data[cluster_labels==e], axis=0)
        std_dev = np.std(data[cluster_labels==e], axis=0)
    
        print(f"Descriptive Statistics for {e}:")
        print("Mean:\n", mean)
        print("Variance:\n", variance)
        print("Standard Deviation:\n", std_dev)
    
    # Plot K-means Clusters
    plt.figure(figsize=(12, 5))
    
    # Plot K-means results
    plt.subplot(1, 2, 1)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', marker='o', s=50, alpha=0.7)
    plt.title("K-means Clustering Results")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    # Plot True Labels
    plt.subplot(1, 2, 2)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=env_idx, cmap='viridis', marker='o', s=50, alpha=0.7)
    plt.title("True Labels")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    plt.tight_layout()
    plt.show()

def cluster_assign():
    datafile = np.load(cluster_path)
    datafile2 = torch.load(path)
    data = datafile['z']
    class_labels = datafile['cluster_label']
    unique_labels = np.unique(class_labels)
    env_idx = datafile['env_idx']
    unique_env_idx = np.unique(env_idx)
    number_cluster = 10
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_env_idx)))
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())

    if True:
        best = None
        score = 0

        for i in range(3):
            print(f"--- {i} ---")
            kmeans = KMeans(n_clusters=number_cluster, random_state=i)
            class_labels = kmeans.fit_predict(data)
            
            # Calculate Silhouette Score
            silhouette_avg = silhouette_score(data, class_labels)
            print(f"Silhouette Score for {number_cluster} clusters: {silhouette_avg:.3f}")
            
            # Calculate Adjusted Rand Index (ARI) if true labels are provided
            ari = adjusted_rand_score(env_idx, class_labels)
            print(f"Adjusted Rand Index (ARI) for {number_cluster} clusters: {ari:.3f}")

            if ari > score:
                best = class_labels
                score = ari

        class_labels = best
        unique_labels = np.unique(class_labels)

    # Initialize t-SNE object
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(data)

    # Plot the result
    plt.figure(figsize=(16, 12))

    for i, label in enumerate(unique_env_idx):
        mask = (env_idx == label)
        plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], label=task_names[label], alpha=0.5)

    plt.title('Task ids')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Create legend
    legend_dict = {task_names[np.unique(env_idx)[i]]: colors[i] for i in range(len(np.unique(env_idx)))}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
    plt.legend(handles=handles, title='Class')

    plt.savefig('Transformer_RNN/graphics/tSNE_plot_task_ids.png', bbox_inches='tight')

    plt.show()

    # Create a second plot for each cluster
    plt.figure(figsize=(16, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = (class_labels == label)
        plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], label=label, alpha=0.5)

    plt.title('t-SNE Visualization by Cluster')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.legend(title='Cluster')

    plt.savefig('Transformer_RNN/graphics/tSNE_plot_clusters.png', bbox_inches='tight')

    plt.show()

    if True:
        rewards = datafile2['rewards'].flatten()
        rewards = rewards / np.max(rewards)
        plt.figure(figsize=(16, 12))

        colors = plt.cm.coolwarm(rewards)

        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors)

        plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='Reward')

        plt.title('Reward Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        plt.savefig('Transformer_RNN/graphics/tSNE_reward_plot.png', bbox_inches='tight')

        plt.show()

def cluster_rnn_assign():
    datafile = torch.load(rnn_path)
    data = datafile['state_emb']
    env_idx = datafile['task']
    unique_env_idx = np.unique(env_idx)
    rewards = datafile['rewards']
    cluster_num = 10

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_env_idx)))
    with open(metadata_path) as f:
        metadata = json.load(f)
    task_names = list(metadata.keys())

    best = None
    score = 0

    for i in range(2):
        print(f"--- {i} ---")
        kmeans = KMeans(n_clusters=cluster_num, random_state=i)
        class_labels = kmeans.fit_predict(data)
        
        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(data, class_labels)
        print(f"Silhouette Score for {cluster_num} clusters: {silhouette_avg:.3f}")
        
        # Calculate Adjusted Rand Index (ARI) if true labels are provided
        ari = adjusted_rand_score(env_idx, class_labels)
        print(f"Adjusted Rand Index (ARI) for {cluster_num} clusters: {ari:.3f}")

        if ari > score:
            best = class_labels
            score = ari

    class_labels = best
    unique_labels = np.unique(class_labels)

    # Initialize t-SNE object
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(data)

    # Plot the result
    plt.figure(figsize=(16, 12))

    for i, label in enumerate(unique_env_idx):
        mask = (env_idx == label)
        plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], label=task_names[label], alpha=0.5)

    plt.title('t-SNE Cluster Visualization', fontsize=22)
    plt.xlabel('t-SNE Component 1', fontsize=19)
    plt.ylabel('t-SNE Component 2', fontsize=19)

    # Create legend
    legend_dict = {task_names[np.unique(env_idx)[i]]: colors[i] for i in range(len(np.unique(env_idx)))}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
    plt.legend(handles=handles, title='Task', fontsize=13, title_fontsize=14)

    plt.savefig('Transformer_RNN/graphics/tSNE_plot_task_ids.png', bbox_inches='tight')

    plt.show()

    # Create a second plot for each cluster
    plt.figure(figsize=(16, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = (class_labels == label)
        plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], color=colors[i], label=label, alpha=0.5)

    plt.title('t-SNE Visualization by Cluster')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.legend(title='Cluster')

    plt.savefig('Transformer_RNN/graphics/tSNE_plot_clusters.png', bbox_inches='tight')

    plt.show()

    if True:
        rewards = rewards / np.max(rewards)
        plt.figure(figsize=(16, 12))

        colors = plt.cm.coolwarm(rewards)

        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], color=colors)

        plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1)), label='Reward')

        plt.title('Reward Visualization', fontsize=22)
        plt.xlabel('t-SNE Component 1', fontsize=19)
        plt.ylabel('t-SNE Component 2', fontsize=19)

        plt.savefig('Transformer_RNN/graphics/tSNE_reward_plot.png', bbox_inches='tight')

        plt.show()

#multiple_samples()
#single_samples()
plot_trajectory_emb()
#cluster_rnn_assign()
#cluster_assign()
#analysis()
