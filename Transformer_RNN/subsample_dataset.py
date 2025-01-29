import torch
from torch.utils.data import Dataset

import numpy as np
import sys, os
import re
import gc

def subsample_buffer(path, buffer_name, save_path, val_save_path):
    # load the replaybuffer
    replay_buffer = DistilledReplayBuffer(
        env_obs_shape=[39], 
        task_obs_shape=(1,), 
        action_shape=[4], 
        capacity=2_000_000,
        batch_size=1, 
        device=torch.device('cuda'), 
        dpmm_batch_size=1,
        normalize_rewards=False
    )
    replay_buffer.load(save_dir=path + buffer_name)
    
    # Reshape and collect all necessary components
    env_obses = np.reshape(replay_buffer.env_obses[:replay_buffer.idx], (-1, 400, 39))
    next_env_obses = np.reshape(replay_buffer.next_env_obses[:replay_buffer.idx], (-1, 400, 39))
    not_dones = np.reshape(replay_buffer.not_dones[:replay_buffer.idx], (-1, 400, 1))
    actions = np.reshape(replay_buffer.actions[:replay_buffer.idx], (-1, 400, 4))
    policy_mu = np.reshape(replay_buffer.policy_mu[:replay_buffer.idx], (-1, 400, 4))
    policy_log_std = np.reshape(replay_buffer.policy_log_std[:replay_buffer.idx], (-1, 400, 4))
    q_target = np.reshape(replay_buffer.q_target[:replay_buffer.idx], (-1, 400, 1))
    rewards = np.reshape(replay_buffer.rewards[:replay_buffer.idx], (-1, 400, 1))
    task_obs = np.reshape(replay_buffer.task_obs[:replay_buffer.idx], (-1, 400, 1))

    if drop_first_percentage > 0:
        samples = np.arange(int(env_obses.shape[0]*drop_first_percentage), env_obses.shape[0])
        
        env_obses = env_obses[samples]
        next_env_obses = next_env_obses[samples]
        not_dones = not_dones[samples]
        actions = actions[samples]
        policy_mu = policy_mu[samples]
        policy_log_std = policy_log_std[samples]
        q_target = q_target[samples]
        rewards = rewards[samples]
        task_obs = task_obs[samples]

    # Downsample to max_num_states: select max_num_states random trajectories
    if leave_last_percantage > 0.:
        split = int((1-leave_last_percantage) * env_obses.shape[0])
        permut = np.concatenate((np.random.permutation(np.arange(split)), np.arange(split, env_obses.shape[0])))
        permut = np.random.permutation(permut)
    else:
        permut = np.random.permutation(np.arange(env_obses.shape[0]))
    val_data_samples = int(env_obses.shape[0] * val_percentage)
    val_samples = permut[:val_data_samples]
    if env_obses.shape[0] > max_num_states:
        #samples = np.unique(np.random.choice(env_obses.shape[0], max_num_states))
        samples = permut[val_data_samples:val_data_samples+max_num_states]
    else:
        samples = permut[val_data_samples:]

    env_obses_train = env_obses[samples]
    next_env_obses_train = next_env_obses[samples]
    not_dones_train = not_dones[samples]
    actions_train = actions[samples]
    policy_mu_train = policy_mu[samples]
    policy_log_std_train = policy_log_std[samples]
    q_target_train = q_target[samples]
    rewards_train = rewards[samples]
    task_obs_train = task_obs[samples]

    env_obses_train = env_obses_train.reshape(-1, 39)
    next_env_obses_train = next_env_obses_train.reshape(-1, 39)
    not_dones_train = not_dones_train.reshape(-1, 1)
    actions_train = actions_train.reshape(-1, 4)
    policy_mu_train = policy_mu_train.reshape(-1, 4)
    policy_log_std_train = policy_log_std_train.reshape(-1, 4)
    q_target_train = q_target_train.reshape(-1, 1)
    rewards_train = rewards_train.reshape(-1, 1)
    task_obs_train = task_obs_train.reshape(-1, 1)
    collected_sub_samples = env_obses_train.shape[0]

    replay_buffer.reset()
    replay_buffer.env_obses[:collected_sub_samples] = env_obses_train
    replay_buffer.next_env_obses[:collected_sub_samples] = next_env_obses_train
    replay_buffer.not_dones[:collected_sub_samples] = not_dones_train
    replay_buffer.actions[:collected_sub_samples] = actions_train
    replay_buffer.policy_mu[:collected_sub_samples] = policy_mu_train
    replay_buffer.policy_log_std[:collected_sub_samples] = policy_log_std_train
    replay_buffer.q_target[:collected_sub_samples] = q_target_train
    replay_buffer.rewards[:collected_sub_samples] = rewards_train
    replay_buffer.task_obs[:collected_sub_samples] = task_obs_train
    replay_buffer.idx = collected_sub_samples

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    replay_buffer.save(save_dir=save_path, size_per_chunk=12_000, num_samples_to_save=-1)
    # validation_dataset
    env_obses_val = env_obses[val_samples]
    next_env_obses_val = next_env_obses[val_samples]
    not_dones_val = not_dones[val_samples]
    actions_val = actions[val_samples]
    policy_mu_val = policy_mu[val_samples]
    policy_log_std_val = policy_log_std[val_samples]
    q_target_val = q_target[val_samples]
    rewards_val = rewards[val_samples]
    task_obs_val = task_obs[val_samples]

    env_obses_val = env_obses_val.reshape(-1, 39)
    next_env_obses_val = next_env_obses_val.reshape(-1, 39)
    not_dones_val = not_dones_val.reshape(-1, 1)
    actions_val = actions_val.reshape(-1, 4)
    policy_mu_val = policy_mu_val.reshape(-1, 4)
    policy_log_std_val = policy_log_std_val.reshape(-1, 4)
    q_target_val = q_target_val.reshape(-1, 1)
    rewards_val = rewards_val.reshape(-1, 1)
    task_obs_val = task_obs_val.reshape(-1, 1)
    collected_sub_samples = env_obses_val.shape[0]

    replay_buffer.reset()
    replay_buffer.env_obses[:collected_sub_samples] = env_obses_val
    replay_buffer.next_env_obses[:collected_sub_samples] = next_env_obses_val
    replay_buffer.not_dones[:collected_sub_samples] = not_dones_val
    replay_buffer.actions[:collected_sub_samples] = actions_val
    replay_buffer.policy_mu[:collected_sub_samples] = policy_mu_val
    replay_buffer.policy_log_std[:collected_sub_samples] = policy_log_std_val
    replay_buffer.q_target[:collected_sub_samples] = q_target_val
    replay_buffer.rewards[:collected_sub_samples] = rewards_val
    replay_buffer.task_obs[:collected_sub_samples] = task_obs_val
    replay_buffer.idx = collected_sub_samples

    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)

    replay_buffer.save(save_dir=val_save_path, size_per_chunk=12_000, num_samples_to_save=-1)
    #

def fix(path, buffer_name, save_path):
    replay_buffer = TransformerReplayBuffer(
        env_obs_shape=[21], 
        task_obs_shape=(1,), 
        action_shape=[4], 
        capacity=2_000_000,
        batch_size=1, 
        device=torch.device('cuda'), 
        dpmm_batch_size=1,
        normalize_rewards=False,
        seq_len=15
    )
    replay_buffer.load(save_dir=path + buffer_name)
    import json
    with open("/home/andi/Desktop/mtrl/metadata/task_embedding/roberta_small/metaworld-all.json") as f:
            metadata = json.load(f)
    metadata_keys = list(metadata.keys())
    distill_names_with_seed = buffer_name.replace("buffer_distill_", "")
    pattern = r'(-\d+)?_seed_\d+$'
    distill_names = re.sub(pattern, '', distill_names_with_seed)
    tasks_obs_to_name = metadata_keys.index(distill_names)
    shape = replay_buffer.task_obs[:replay_buffer.idx//400].shape
    replay_buffer.task_obs[:replay_buffer.idx//400] = np.full(shape, tasks_obs_to_name)
    
    replay_buffer.last_save = 0
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    replay_buffer.save(save_dir=save_path, size_per_chunk=12_000, num_samples_to_save=-1)

def add_task_encoding(encoder, prediction_head_cls, bnpy_mod, buffer_path, buffer_name, save_path):
    replay_buffer_distilled = DistilledReplayBuffer(
        env_obs_shape=[39], 
        task_obs_shape=(1,), 
        action_shape=[4], 
        capacity=2_000_000,
        batch_size=1, 
        device=torch.device('cuda'), 
        dpmm_batch_size=1,
        normalize_rewards=False
    )
    replay_buffer_distilled.load(save_dir=buffer_path)

    replay_buffer_new = TransformerReplayBuffer(
        env_obs_shape=[21], 
        task_obs_shape=(1,), 
        action_shape=[4], 
        capacity=2_000_000,
        batch_size=1, 
        device=torch.device('cuda'), 
        dpmm_batch_size=1,
        normalize_rewards=False,
        seq_len=sequence_length,
        task_encoding_shape=5,
        compressed_state=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    prediction_head_cls = prediction_head_cls.to(device)
    import json
    with open("/home/andi/Desktop/mtrl/metadata/task_embedding/roberta_small/metaworld-all.json") as f:
            metadata = json.load(f)
    metadata_keys = list(metadata.keys())
    buffer_name = buffer_name.replace("buffer_distill_tmp_", "")
    distill_names_with_seed = buffer_name.replace("buffer_distill_", "")
    pattern = r'(-\d+)?_seed_\d+$'
    distill_names = re.sub(pattern, '', distill_names_with_seed)
    tasks_obs_to_name = metadata_keys.index(distill_names)
    for i in range(replay_buffer_distilled.idx):
        last_x_samples = min(i%400, sequence_length-1)
        states = torch.tensor(replay_buffer_distilled.env_obses[i-last_x_samples:i+1]).to(device=device)
        rewards = torch.tensor(replay_buffer_distilled.rewards[i-last_x_samples:i]).to(device=device)
        actions = torch.tensor(replay_buffer_distilled.actions[i-last_x_samples:i]).to(device=device)
        encoding = encoder(states[:,:21][None], actions[None], rewards[None])[-1]
        encoding = prediction_head_cls(encoding)
        task_encoding, _ = bnpy_mod.cluster(encoding)
        replay_buffer_new.add(
            env_obs=replay_buffer_distilled.env_obses[i],
            next_env_obs=replay_buffer_distilled.next_env_obses[i],
            action=replay_buffer_distilled.actions[i],
            reward=replay_buffer_distilled.rewards[i], 
            done=not replay_buffer_distilled.not_dones[i],
            task_obs=tasks_obs_to_name,
            encoding=task_encoding.cpu().numpy(),
            q_value=replay_buffer_distilled.q_target[i],
            mu=replay_buffer_distilled.policy_mu[i],
            log_std=replay_buffer_distilled.policy_log_std[i]
        )
        if i % 10_000 == 0:
            print(f"[{i}/{replay_buffer_distilled.idx}]")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    replay_buffer_new.save(save_dir=save_path, size_per_chunk=12_000, num_samples_to_save=-1)

def collect_only_last_buffer(path, buffer_name, save_path):
    # load the replaybuffer
    replay_buffer = DistilledReplayBuffer(
        env_obs_shape=[39], 
        task_obs_shape=(1,), 
        action_shape=[4], 
        capacity=2_000_000,
        batch_size=1, 
        device=torch.device('cuda'), 
        dpmm_batch_size=1,
        normalize_rewards=False
    )
    replay_buffer.load(save_dir=path + buffer_name)
    
    # Reshape and collect all necessary components
    env_obses = np.reshape(replay_buffer.env_obses[:replay_buffer.idx], (-1, 400, 39))
    next_env_obses = np.reshape(replay_buffer.next_env_obses[:replay_buffer.idx], (-1, 400, 39))
    not_dones = np.reshape(replay_buffer.not_dones[:replay_buffer.idx], (-1, 400, 1))
    actions = np.reshape(replay_buffer.actions[:replay_buffer.idx], (-1, 400, 4))
    policy_mu = np.reshape(replay_buffer.policy_mu[:replay_buffer.idx], (-1, 400, 4))
    policy_log_std = np.reshape(replay_buffer.policy_log_std[:replay_buffer.idx], (-1, 400, 4))
    q_target = np.reshape(replay_buffer.q_target[:replay_buffer.idx], (-1, 400, 1))
    rewards = np.reshape(replay_buffer.rewards[:replay_buffer.idx], (-1, 400, 1))
    task_obs = np.reshape(replay_buffer.task_obs[:replay_buffer.idx], (-1, 400, 1))
    
    env_obses = env_obses[-keep_last_x_step//400:]
    next_env_obses = next_env_obses[-keep_last_x_step//400:]
    not_dones = not_dones[-keep_last_x_step//400:]
    actions = actions[-keep_last_x_step//400:]
    policy_mu = policy_mu[-keep_last_x_step//400:]
    policy_log_std = policy_log_std[-keep_last_x_step//400:]
    q_target = q_target[-keep_last_x_step//400:]
    rewards = rewards[-keep_last_x_step//400:]
    task_obs = task_obs[-keep_last_x_step//400:]

    env_obses = env_obses.reshape(-1, 39)
    next_env_obses = next_env_obses.reshape(-1, 39)
    not_dones = not_dones.reshape(-1, 1)
    actions = actions.reshape(-1, 4)
    policy_mu = policy_mu.reshape(-1, 4)
    policy_log_std = policy_log_std.reshape(-1, 4)
    q_target = q_target.reshape(-1, 1)
    rewards = rewards.reshape(-1, 1)
    task_obs = task_obs.reshape(-1, 1)
    collected_sub_samples = env_obses.shape[0]

    replay_buffer.reset()
    replay_buffer.env_obses[:collected_sub_samples] = env_obses
    replay_buffer.next_env_obses[:collected_sub_samples] = next_env_obses
    replay_buffer.not_dones[:collected_sub_samples] = not_dones
    replay_buffer.actions[:collected_sub_samples] = actions
    replay_buffer.policy_mu[:collected_sub_samples] = policy_mu
    replay_buffer.policy_log_std[:collected_sub_samples] = policy_log_std
    replay_buffer.q_target[:collected_sub_samples] = q_target
    replay_buffer.rewards[:collected_sub_samples] = rewards
    replay_buffer.task_obs[:collected_sub_samples] = task_obs
    replay_buffer.idx = collected_sub_samples

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    replay_buffer.save(save_dir=save_path, size_per_chunk=12_000, num_samples_to_save=-1)


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join('..', 'mtrl')))
    from mtrl.col_replay_buffer import DistilledReplayBuffer
    from mtrl.transformer_replay_buffer import TransformerReplayBuffer
    from mtrl.replay_buffer import ReplayBuffer
    from mtrl.agent.components.transformer_trajectory_encoder import RepresentationEncoderTransformer, ClsPredictionHead, Bnpy_model

    encoder = RepresentationEncoderTransformer(
        state_dim=21,
        action_dim=4,
        d_model=32, #28
        nhead=8,
        dim_feedforward=128,
        nlayers=3,
        sequence_len=20, #15
        dropout=0.,
        model_path="/home/andi/Desktop/mtrl/Transformer_RNN/checkpoints/representation_cls_transformer_checkpoint.pth",
    )
    prediction_head_cls = ClsPredictionHead(
        32, #28
        6, #5 
        model_path="/home/andi/Desktop/mtrl/Transformer_RNN/checkpoints/representation_cls_transformer_checkpoint.pth"
    )
    bnpy_mod = Bnpy_model(bnpy_load_path="/home/andi/Desktop/mtrl/Transformer_RNN/bnpy_save/save", device="cuda" if torch.cuda.is_available() else "cpu")

    path_data = '/home/andi/Desktop/mtrl/Transformer_RNN/dataset/'
    safe_path = 'Transformer_RNN/subsampled_datasets/train/'
    val_safe_path = 'Transformer_RNN/subsampled_datasets/validation/'
    subdicts = ['tmp4/'] # 'new_init/' 'distill/' 'expert/' 'kuka/' 'saywer/' 'distill_all/'
    tasks = ['pick-place']
    all_subsample = True
    iterate = True

    seed = 0
    np.random.seed(seed)
    max_num_states = 600
    max_sequence_length = 400
    keep_last_x_step = 200_000
    val_percentage = 0.05
    num_chunks = 0
    drop_first_percentage = 0. # 0.6
    leave_last_percantage = 0.33
    drop_minimum_samples = 1000
    sequence_length = 15

    # create training dataset
    if iterate:
        for sub in subdicts:        
            distill_buffer_names = os.listdir(path_data + sub)
            pattern = r'(-\d+)?_seed_\d+$'
            distill_names = [re.sub(pattern, '', name.replace("buffer_distill_", "")) for name in distill_buffer_names]

            for buffer in distill_buffer_names:
                if any(task in buffer for task in tasks) or all_subsample:
                    #subsample_buffer(path_data + sub, buffer, safe_path + buffer, val_safe_path + buffer)
                    #collect_only_last_buffer(path_data + sub, buffer, safe_path + buffer)
                    add_task_encoding(encoder, prediction_head_cls, bnpy_mod, path_data + sub + buffer, buffer, safe_path + buffer)