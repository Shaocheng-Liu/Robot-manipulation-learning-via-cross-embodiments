import torch
from torch.utils.data import Dataset

import numpy as np
import sys, os
import re
import gc

import re
from pathlib import Path



class TFDataset(Dataset):
    def __init__(self, states, actions, rewards, policy_mu, task_obs, task_arm, max_seq_len, sequence_length):
        """
        Args:
            states (list of tensors): A list where each entry is a tensor representing the state at a time step.
            actions (list of tensors): A list where each entry is a tensor representing the action taken at a time step.
            rewards (list of tensors): A list where each entry is a tensor representing the reward received at a time step.
            sequence_length (int): The number of time steps to return for each sample.
            tra x max_time_steps
        """
        self.states = states
        self.actions = actions
        self.policy_mu = policy_mu
        self.rewards = rewards
        self.task_obs = task_obs
        self.task_arm = task_arm
        self.max_seq_len = max_seq_len
        self.sequence_length = sequence_length
        self.avaible_length = self.max_seq_len - self.sequence_length + 1
        self.unique_task_obs = np.unique(task_obs)

        #TODO: self.min_value = 0
        #TODO: self.max_value = 0

        # Ensure that the input lists have the same length
        assert len(self.states) == len(self.actions) == len(self.rewards), "All input lists must have the same length."
        assert len(self.task_obs) == len(self.task_arm), "Task_obs and task arm must match"
        assert len(self.states) >= self.sequence_length, "Not enough data to create even one sequence."

    def __len__(self):
        # The number of valid sequences
        return len(self.states) * self.avaible_length

    def __getitem__(self, idx):
        state_sequence = self.states[idx//self.avaible_length][idx%self.avaible_length:(idx%self.avaible_length+self.sequence_length)]
        action_sequence = self.actions[idx//self.avaible_length][idx%self.avaible_length:(idx%self.avaible_length+self.sequence_length)]
        policy_mu_sequence = self.policy_mu[idx//self.avaible_length][idx%self.avaible_length:(idx%self.avaible_length+self.sequence_length)]
        reward_sequence = self.rewards[idx//self.avaible_length][idx%self.avaible_length:(idx%self.avaible_length+self.sequence_length)]
        task_obs = self.task_obs[idx//self.avaible_length]
        task_arm = self.task_arm[idx//self.avaible_length]

        return state_sequence, action_sequence, policy_mu_sequence, reward_sequence, task_obs, task_arm
    
    def save(self, file_path, start_chunk):
        chunk_size = 2_500  # Adjust based on your memory constraints
        num_chunks = len(self.states) // chunk_size + 1
        for i in range(num_chunks):
            chunk_path = f"{file_path}_chunk_{start_chunk+i}"
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            print(f"saved {chunk_path}")
            os.makedirs(safe_path, exist_ok=True)
            torch.save({
                "states": self.states[start_idx:end_idx],
                "actions": self.actions[start_idx:end_idx],
                "policy_mu": self.policy_mu[start_idx:end_idx],
                "rewards": self.rewards[start_idx:end_idx],
                "task_obs": self.task_obs[start_idx:end_idx],
                "task_arm": self.task_arm[start_idx:end_idx]
            }, chunk_path)
            gc.collect()
        return num_chunks
    
    @classmethod
    def load(cls, file_path, max_sequence_length=400, sequence_length=20):
        data = {"states": [], "actions": [], "policy_mu": [], "rewards": [], "task_obs": [], "task_arm": []}
        i = 0
        while os.path.isfile(f"{file_path}_chunk_{i}"):
            chunk_data = torch.load(f"{file_path}_chunk_{i}")
            data["states"].extend(chunk_data["states"])
            data["actions"].extend(chunk_data["actions"])
            data["policy_mu"].extend(chunk_data["policy_mu"])
            data["rewards"].extend(chunk_data["rewards"])
            data["task_obs"].extend(chunk_data["task_obs"])
            data["task_arm"].extend(chunk_data["task_arm"])
            i += 1
        return cls(np.stack(data['states']), np.stack(data['actions']), np.stack(data['policy_mu']), np.stack(data['rewards']), np.stack(data["task_obs"]), np.stack(data["task_arm"]), max_sequence_length, sequence_length)
    
def load_and_preprocess_buffer(path, buffer_name, arm_count):
    # load the replaybuffer
    replay_buffer = DistilledReplayBuffer(
        env_obs_shape=[39], 
        task_obs_shape=(1,), 
        action_shape=[4], 
        capacity=2_000_000,
        batch_size=1, 
        device=torch.device('cuda'), 
        normalize_rewards=False
    )
    replay_buffer.load(save_dir=path + buffer_name)
    
    # Reshape and collect all necessary components
    states = np.reshape(replay_buffer.env_obses[:replay_buffer.idx], (-1, 400, 39))
    states = np.concatenate((states[:,:,:18], states[:,:,36:]), axis=-1)
    rewards = np.reshape(replay_buffer.rewards[:replay_buffer.idx], (-1, 400, 1))
    actions = np.reshape(replay_buffer.actions[:replay_buffer.idx], (-1, 400, 4))
    policy_mu = np.reshape(replay_buffer.policy_mu[:replay_buffer.idx], (-1, 400, 4))
    task_obs = np.reshape(replay_buffer.task_obs[:replay_buffer.idx], (-1, 400, 1))[:, 0].flatten()
    task_arm = np.full(task_obs.shape, arm_count)

    if drop_first_percentage > 0 and states.shape[0] > drop_minimum_samples:
        samples = np.arange(int(states.shape[0]*drop_first_percentage), states.shape[0])
        states = states[samples]
        rewards = rewards[samples]
        actions = actions[samples]
        policy_mu = policy_mu[samples]
        task_obs = task_obs[samples]
        task_arm = task_arm[samples]

    # Downsample to max_num_states: select max_num_states random trajectories
    if states.shape[0] > max_num_states:
        samples = np.random.choice(states.shape[0], max_num_states)

        states = states[samples]
        rewards = rewards[samples]
        actions = actions[samples]
        policy_mu = policy_mu[samples]
        task_obs = task_obs[samples]
        task_arm = task_arm[samples]

    return states, rewards, actions, policy_mu, task_obs, task_arm


def parse_robot_and_task(buffer_name: str):
    base = Path(buffer_name).stem.replace("buffer_distill_", "")
    base = re.sub(r'(-\d+)?_seed_\d+$', '', base)
    robot, task = base.split('_', 1)
    return robot, task

######################################################################

import torch, numpy as np, builtins
def log_blue(msg):
    print(f"\033[34m{msg}\033[0m")

log_blue("[SAVE] Replay buffer to ./save/path")

_orig_save = torch.save
def save_with_log(obj, f, *args, **kwargs):
    log_blue(f"[TORCH SAVE] → {f}")
    return _orig_save(obj, f, *args, **kwargs)
torch.save = save_with_log

_orig_load = torch.load
def load_with_log(f, *args, **kwargs):
    log_blue(f"[TORCH LOAD] ← {f}")
    return _orig_load(f, *args, **kwargs)
torch.load = load_with_log

# Similarly for np.save / np.load
np_save = np.save
np_load = np.load

def logged_np_save(file, arr, *args, **kwargs):
    log_blue(f"[NP SAVE] → {file}")
    return np_save(file, arr, *args, **kwargs)
np.save = logged_np_save

def logged_np_load(file, *args, **kwargs):
    log_blue(f"[NP LOAD] ← {file}")
    return np_load(file, *args, **kwargs)
np.load = logged_np_load


######################################################################

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join('..', 'mtrl')))
    from mtrl.col_replay_buffer import DistilledReplayBuffer
    project_root = os.environ.get("PROJECT_ROOT")
    path_data = project_root + "/Transformer_RNN/dataset/"
    safe_path = 'Transformer_RNN/decision_tf_dataset/'
    subdicts = ['train/','validation/' ]

    seed = 0
    np.random.seed(seed)
    sequence_length = 5
    max_num_states = 2200
    max_sequence_length = 400
    num_chunks = 0
    drop_first_percentage = 0.25
    drop_minimum_samples = 1000

    print(f"Creating dataset with max samples {max_num_states} and seq length {sequence_length} and seed {seed}")

    arm_count = 0

    robot2id, next_rid = {}, 0
    for sub in subdicts:
        for name in os.listdir(path_data + sub):
            robot, _ = parse_robot_and_task(name)
            if robot not in robot2id:
                robot2id[robot] = next_rid; next_rid += 1

    # create training dataset
    for sub in subdicts:
        states_arr, rewards_arr, actions_arr, policy_mu_arr, task_obs_arr, task_arm_arr = [], [], [], [], [], []
        
        distill_buffer_names = os.listdir(path_data + sub)
        pattern = r'(-\d+)?_seed_\d+$'
        distill_names = [re.sub(pattern, '', name.replace("buffer_distill_", "")) for name in distill_buffer_names]

        for buffer in distill_buffer_names:
            robot, _ = parse_robot_and_task(buffer)
            arm_id = robot2id[robot]
            states, rewards, actions, policy_mu, task_obs, task_arm = load_and_preprocess_buffer(path_data + sub, buffer, arm_id)
            states_arr.append(states)
            rewards_arr.append(rewards)
            actions_arr.append(actions)
            policy_mu_arr.append(policy_mu)
            task_obs_arr.append(task_obs) 
            task_arm_arr.append(task_arm)
        
        gc.collect()

        states_arr = np.concatenate(states_arr)
        rewards_arr = np.concatenate(rewards_arr)
        actions_arr = np.concatenate(actions_arr)
        policy_mu_arr = np.concatenate(policy_mu_arr)
        task_obs_arr = np.concatenate(task_obs_arr)
        task_arm_arr = np.concatenate(task_arm_arr)

        dataset = TFDataset(states_arr, actions_arr, policy_mu_arr, rewards_arr, task_obs_arr, task_arm_arr, max_sequence_length, sequence_length=sequence_length)
        print("Saving dataset...")
        num_chunks += dataset.save(safe_path, num_chunks)
        print(f"Saved dataset of length {len(states_arr)}")
        
        arm_count += 1
        gc.collect()
