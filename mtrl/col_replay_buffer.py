# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mtrl.utils.types import TensorType


@dataclass
class DistilledReplayBufferSample:
    __slots__ = [
        "env_obs",
        "action",
        "reward",
        "next_env_obs",
        "not_done",
        "task_obs",
        "buffer_index",
        "policy_mu",
        "policy_log_std",
        "q_target",
    ]
    env_obs: TensorType
    action: TensorType
    reward: TensorType
    next_env_obs: TensorType
    not_done: TensorType
    task_obs: TensorType
    buffer_index: TensorType
    policy_mu: TensorType
    policy_log_std: TensorType
    q_target: TensorType


class DistilledReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(
        self, env_obs_shape, task_obs_shape, action_shape, capacity, batch_size, device, normalize_rewards
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.normalize_rewards = normalize_rewards

        # the proprioceptive env_obs is stored as float32, pixels env_obs as uint8
        env_obs_dtype = np.float32 if len(env_obs_shape) == 1 else np.uint8
        task_obs_dtype = np.int64

        self.env_obses = np.empty((capacity, *env_obs_shape), dtype=env_obs_dtype)
        self.next_env_obses = np.empty((capacity, *env_obs_shape), dtype=env_obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.task_obs = np.empty((capacity, *task_obs_shape), dtype=task_obs_dtype)
        self.policy_mu = np.empty((capacity, *action_shape), dtype=np.float32)
        self.policy_log_std = np.empty((capacity, *action_shape), dtype=np.float32)
        self.q_target = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.max_reward = 0
        self.min_reward = 0

    def is_empty(self):
        return self.idx == 0

    def add(self, env_obs, action, reward, next_env_obs, done, task_obs, q_value, mu, log_std):
        np.copyto(self.env_obses[self.idx], env_obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_env_obses[self.idx], next_env_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.task_obs[self.idx], task_obs)
        np.copyto(self.policy_mu[self.idx], mu)
        np.copyto(self.policy_log_std[self.idx], log_std)
        np.copyto(self.q_target[self.idx], q_value)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_array(self, env_obs, action, reward, next_env_obs, done, task_obs, q_value, mu, log_std, size):
        np.copyto(self.env_obses[self.idx:self.idx+size], env_obs)
        np.copyto(self.actions[self.idx:self.idx+size], action)
        np.copyto(self.rewards[self.idx:self.idx+size], reward)
        np.copyto(self.next_env_obses[self.idx:self.idx+size], next_env_obs)
        np.copyto(self.not_dones[self.idx:self.idx+size], np.logical_not(done))
        np.copyto(self.task_obs[self.idx:self.idx+size], task_obs)
        np.copyto(self.q_target[self.idx:self.idx+size], q_value)
        np.copyto(self.policy_mu[self.idx:self.idx+size], mu)
        np.copyto(self.policy_log_std[self.idx:self.idx+size], log_std)

        if self.idx + size > self.capacity:
            raise NotImplementedError("Exceeded capacity")
        self.idx = (self.idx + size) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, index=None) -> DistilledReplayBufferSample:
        b_size = self.batch_size
        
        if index is None:
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=b_size
            )
        else:
            idxs = index

        env_obses = torch.as_tensor(self.env_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_env_obses = torch.as_tensor(
            self.next_env_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        env_indices = torch.as_tensor(self.task_obs[idxs], device=self.device)
        mus = torch.as_tensor(self.policy_mu[idxs], device=self.device)
        log_stds = torch.as_tensor(self.policy_log_std[idxs], device=self.device)
        q_targets = torch.as_tensor(self.q_target[idxs], device=self.device)

        return DistilledReplayBufferSample(
            env_obses, actions, rewards, next_env_obses, not_dones, env_indices, idxs, mus, log_stds, q_targets
        )
    
    def sample_trajectories(self, seq_len, max_timestep, index=None):      
        if index is None:
            start_idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )
        else:
            start_idxs = index
        
        segments = np.full((self.batch_size, seq_len, 44), -1, dtype=np.float32)
        valid_lengths = np.minimum(seq_len, (start_idxs+1) % max_timestep)

        for i in range(self.batch_size):
            segments[i, seq_len-valid_lengths[i]:] = np.concatenate((self.env_obses[start_idxs[i]+1 - valid_lengths[i]:start_idxs[i]+1],
                                                             self.actions[start_idxs[i]+1 - valid_lengths[i]:start_idxs[i]+1],
                                                             self.rewards[start_idxs[i]+1 - valid_lengths[i]:start_idxs[i]+1]), axis=1)

        segments = torch.as_tensor(segments, device=self.device).float()
        env_indices = torch.as_tensor(self.task_obs[start_idxs], device=self.device)
        mus = torch.as_tensor(self.policy_mu[start_idxs], device=self.device)
        log_stds = torch.as_tensor(self.policy_log_std[start_idxs], device=self.device)
        q_targets = torch.as_tensor(self.q_target[start_idxs], device=self.device)

        return segments, mus, log_stds, q_targets, env_indices

    def delete_from_filesystem(self, dir_to_delete_from: str):
        for filename in os.listdir(dir_to_delete_from):
            file_path = os.path.join(dir_to_delete_from, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print(f"Deleted files from: {dir_to_delete_from}")

    def save(self, save_dir, size_per_chunk: int, num_samples_to_save: int):
        if self.idx == self.last_save:
            return
        if num_samples_to_save == -1:
            # Save the entire replay buffer
            self._save_all(
                save_dir=save_dir,
                size_per_chunk=size_per_chunk,
            )
        else:
            raise NotImplementedError("TODO: save functionality is not implemented yet")
        
            if num_samples_to_save > self.idx:
                num_samples_to_save = self.idx
                replay_buffer_to_save = self
            else:
                replay_buffer_to_save = self._sample_a_replay_buffer(
                    num_samples=num_samples_to_save
                )
                replay_buffer_to_save.idx = num_samples_to_save
                replay_buffer_to_save.last_save = 0
            backup_dir_path = Path(f"{save_dir}_bk")
            if not backup_dir_path.exists():
                backup_dir_path.mkdir()
            replay_buffer_to_save._save_all(
                save_dir=str(backup_dir_path),
                size_per_chunk=size_per_chunk,
            )
            replay_buffer_to_save.delete_from_filesystem(dir_to_delete_from=save_dir)
            backup_dir_path.rename(save_dir)
        self.last_save = self.idx

    def _save_all(self, save_dir, size_per_chunk: int):
        if self.idx == self.last_save:
            return
        if self.last_save == self.capacity:
            self.last_save = 0
        if self.idx > self.last_save:
            self._save_payload(
                save_dir=save_dir,
                start_idx=self.last_save,
                end_idx=self.idx,
                size_per_chunk=size_per_chunk,
            )
        else:
            self._save_payload(
                save_dir=save_dir,
                start_idx=self.last_save,
                end_idx=self.capacity,
                size_per_chunk=size_per_chunk,
            )
            self._save_payload(
                save_dir=save_dir,
                start_idx=0,
                end_idx=self.idx,
                size_per_chunk=size_per_chunk,
            )
        self.last_save = self.idx

    def _save_payload(
        self, save_dir: str, start_idx: int, end_idx: int, size_per_chunk: int
    ):
        while True:
            if size_per_chunk > 0:
                current_end_idx = min(start_idx + size_per_chunk, end_idx)
            else:
                current_end_idx = end_idx
            self._save_payload_chunk(
                save_dir=save_dir, start_idx=start_idx, end_idx=current_end_idx
            )
            if current_end_idx == end_idx:
                break
            start_idx = current_end_idx

    def _save_payload_chunk(self, save_dir: str, start_idx: int, end_idx: int):
        path = os.path.join(save_dir, f"{start_idx}_{end_idx-1}.pt")
        payload = [
            self.env_obses[start_idx:end_idx],
            self.next_env_obses[start_idx:end_idx],
            self.actions[start_idx:end_idx],
            self.rewards[start_idx:end_idx],
            self.not_dones[start_idx:end_idx],
            self.task_obs[start_idx:end_idx],
            self.policy_mu[start_idx:end_idx],
            self.policy_log_std[start_idx:end_idx],
            self.q_target[start_idx:end_idx],
        ]
        print(f"Saving replay buffer at {path}")
        torch.save(payload, path)

    def min_max_normalize(self, tensor, min_value, max_value):
        return (tensor - min_value) / (max_value - min_value)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chunks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        start = 0
        for chunk in chunks:
            path = os.path.join(save_dir, chunk)
            try:
                payload = torch.load(path)
                end = start + payload[0].shape[0]
                if end > self.capacity:
                    # this condition is added for resuming some very old experiments.
                    # This condition should not be needed with the new experiments
                    # and should be removed going forward.
                    select_till_index = payload[0].shape[0] - (end - self.capacity)
                    end = start + select_till_index
                else:
                    select_till_index = payload[0].shape[0]
                self.env_obses[start:end] = payload[0][:select_till_index]
                self.next_env_obses[start:end] = payload[1][:select_till_index]
                self.actions[start:end] = payload[2][:select_till_index]
                self.rewards[start:end] = payload[3][:select_till_index]
                self.not_dones[start:end] = payload[4][:select_till_index]
                self.task_obs[start:end] = payload[5][:select_till_index]
                self.policy_mu[start:end] = payload[6][:select_till_index]
                self.policy_log_std[start:end] = payload[7][:select_till_index]
                self.q_target[start:end] = payload[8][:select_till_index]
                self.idx = end # removed - 1
                start = end
                print(f"Loaded replay buffer from path: {path})")
            except EOFError as e:
                print(
                    f"Skipping loading replay buffer from path: {path} due to error: {e}"
                )
        self.last_save = self.idx
        if self.normalize_rewards:
            self.max_reward = np.max(self.rewards[:self.idx])
            self.min_reward = np.min(self.rewards[:self.idx])
            self.rewards[:self.idx] = self.min_max_normalize(self.rewards[:self.idx], self.min_reward, self.max_reward)
        # self.delete_from_filesystem(dir_to_delete_from=save_dir)

    def load_multiple_buffer(self, save_dirs):
        start = 0
        for save_dir in save_dirs:
            chunks = os.listdir(save_dir)
            chunks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
            for chunk in chunks:
                path = os.path.join(save_dir, chunk)
                try:
                    payload = torch.load(path)
                    end = start + payload[0].shape[0]
                    if end > self.capacity:
                        # this condition is added for resuming some very old experiments.
                        # This condition should not be needed with the new experiments
                        # and should be removed going forward.
                        select_till_index = payload[0].shape[0] - (end - self.capacity)
                        end = start + select_till_index
                    else:
                        select_till_index = payload[0].shape[0]
                    self.env_obses[start:end] = payload[0][:select_till_index]
                    self.next_env_obses[start:end] = payload[1][:select_till_index]
                    self.actions[start:end] = payload[2][:select_till_index]
                    self.rewards[start:end] = payload[3][:select_till_index]
                    self.not_dones[start:end] = payload[4][:select_till_index]
                    self.task_obs[start:end] = payload[5][:select_till_index]
                    self.policy_mu[start:end] = payload[6][:select_till_index]
                    self.policy_log_std[start:end] = payload[7][:select_till_index]
                    self.q_target[start:end] = payload[8][:select_till_index]
                    self.idx = end - 1
                    start = end
                    print(f"Loaded replay buffer from path: {path})")
                except EOFError as e:
                    print(
                        f"Skipping loading replay buffer from path: {path} due to error: {e}"
                    )
        self.last_save = self.idx

    def reset(self):
        self.idx = 0
        self.last_save = 0
