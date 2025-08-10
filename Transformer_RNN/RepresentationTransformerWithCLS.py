import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import math
from dataset_tf import TFDataset
import os
from InfoNceLoss import InfoNCE
from torch.utils.tensorboard import SummaryWriter

import bnpy
from bnpy.data.XData import XData
from itertools import cycle
from matplotlib import pylab
import shutil


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TrainablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10):
        """
        Initialize the Trainable Positional Encoding layer.

        Args:
        - max_seq_len (int): The maximum sequence length.
        - d_model (int): The dimension of the model (the size of the embeddings).
        """
        super(TrainablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a learnable parameter for positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Trainable Positional Encoding layer.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
        - torch.Tensor: Output tensor with added positional encoding of shape (batch_size, seq_len, d_model).
        """
        # Add the positional encoding up to the sequence length
        x = x + self.positional_encoding[:x.size(0), :]

        return self.dropout(x)

class EncoderOnlyTransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, state_dim, action_dim, d_model, nhead, dim_feedforward, nlayers, sequence_len, device, dropout=0.25):
        super(EncoderOnlyTransformerModel, self).__init__()
        self.pos_encoder = TrainablePositionalEncoding(d_model, dropout, sequence_len)

        self.state_emb = nn.Linear(state_dim, d_model)
        self.action_emb = nn.Linear(action_dim, d_model)
        self.reward_emb = nn.Linear(1, d_model)

        self.state_norm = nn.LayerNorm(d_model)
        self.action_norm = nn.LayerNorm(d_model)
        self.reward_norm = nn.LayerNorm(d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.device = device
        self.d_model = d_model

        #self.special_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.special_token = nn.Parameter(torch.full((1, 1, d_model), -2.))
    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def forward(self, states_src, actions_src, rewards_src, src_mask=None,):
        batch_size, seq_len = actions_src.shape[0], actions_src.shape[1]

        state_token_src = self.state_norm(self.state_emb(states_src)) # * torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        action_token_src = self.action_norm(self.action_emb(actions_src)) # * torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        reward_token_src = self.reward_norm(self.reward_emb(rewards_src)) # * torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))

        state_token_src = self.pos_encoder(state_token_src.permute(1, 0, 2))
        reward_token_src = self.pos_encoder(reward_token_src.permute(1, 0, 2))
        action_token_src = self.pos_encoder(action_token_src.permute(1, 0, 2))
        
        special_token = self.special_token.expand(-1, batch_size, -1)

        src = torch.stack((action_token_src, reward_token_src, state_token_src[1:]), dim=0).transpose(0,1).reshape(3*seq_len, -1, self.d_model) # shape: 30, batchsize, hidden size
        src = torch.cat((state_token_src[0][None], src, special_token), dim=0)
        
        output = self.transformer_encoder(src, mask=src_mask)
        return output
    
class PredictionHead(nn.Module):
    def __init__(self, encoding_dim, output_dim):
        super(PredictionHead, self).__init__()
        self.linear = nn.Linear(encoding_dim, output_dim)

    def forward(self, encoding):
        return self.linear(encoding)
    
class ClassificationHead(nn.Module):
    def __init__(self, encoding_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(encoding_dim, num_classes)


    def forward(self, encoding):
        logits = self.linear(encoding)
        prob_vec = F.softmax(logits, dim=1)
        return prob_vec
    

def train_loop(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, prediction_head_cls, opt, loss_fn, contrastive_loss_fn, data_loader):
    encoder.train()
    prediction_head_state.train()
    prediction_head_action.train()
    prediction_head_reward.train()
    prediction_head_cls.train()
    recon_loss = 0
    contra_loss = 0
    counter = 0
    len_dataloader = len(data_loader)

    state_indices = torch.arange(0, sequence_len*3, 3) 
    action_indices = torch.arange(1, (sequence_len-1)*3, 3)
    reward_indices = torch.arange(2, (sequence_len-1)*3, 3)
    
    for batch in data_loader:
        # select feature for the input and output
        states, actions, _, rewards, task_ind, task_arm = batch
        states = states.to(device)
        actions = actions[:, :-1].to(device)
        rewards = rewards[:, :-1].to(device)
        task_ind = task_ind.to(device)
        task_arm = task_arm.to(device)

        state_inp, state_inp_mask, action_inp, action_inp_mask, reward_inp, reward_inp_mask = mask_tokens(states, actions, rewards)
        
        # forward pass trough the network
        encoding = encoder(state_inp, action_inp, reward_inp)
        encoding_state, encoding_action, encoding_reward, encoding_cls_token = encoding[state_indices], encoding[action_indices], encoding[reward_indices], encoding[-1]
        pred_state = prediction_head_state(encoding_state).permute(1, 0, 2)
        pred_action = prediction_head_action(encoding_action).permute(1, 0, 2)
        pred_reward = prediction_head_reward(encoding_reward).permute(1, 0, 2)
        cls_token = prediction_head_cls(encoding_cls_token)

        pred_state, state_tar = pred_state[state_inp_mask], states[state_inp_mask]
        pred_action, action_tar = pred_action[action_inp_mask], actions[action_inp_mask]
        pred_reward, reward_tar = pred_reward[reward_inp_mask], rewards[reward_inp_mask]

        # positive samples
        state_inp_pos, state_inp_mask_pos, action_inp_pos, action_inp_mask_pos, reward_inp_pos, reward_inp_mask_pos = mask_tokens(states, actions, rewards)
        
        # forward pass trough the network
        encoding_pos = encoder(state_inp_pos, action_inp_pos, reward_inp_pos)
        encoding_state_pos, encoding_action_pos, encoding_reward_pos, encoding_cls_token_pos = encoding_pos[state_indices], encoding_pos[action_indices], encoding_pos[reward_indices], encoding_pos[-1]
        pred_state_pos = prediction_head_state(encoding_state_pos).permute(1, 0, 2)
        pred_action_pos = prediction_head_action(encoding_action_pos).permute(1, 0, 2)
        pred_reward_pos = prediction_head_reward(encoding_reward_pos).permute(1, 0, 2)
        cls_token_pos = prediction_head_cls(encoding_cls_token_pos)

        pred_state_pos, state_tar_pos = pred_state_pos[state_inp_mask_pos], states[state_inp_mask_pos]
        pred_action_pos, action_tar_pos = pred_action_pos[action_inp_mask_pos], actions[action_inp_mask_pos]
        pred_reward_pos, reward_tar_pos = pred_reward_pos[reward_inp_mask_pos], rewards[reward_inp_mask_pos]

        # calculate the state-/action-/reward-loss independently
        rec_loss = loss_fn(pred_state, state_tar) + loss_fn(pred_action, action_tar) + loss_fn(pred_reward, reward_tar) + \
                     loss_fn(pred_state_pos, state_tar_pos) + loss_fn(pred_action_pos, action_tar_pos) + loss_fn(pred_reward_pos, reward_tar_pos)
        if not use_own_contrastive_loss:
            contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos)
        elif use_task_arm:
            contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos, task_arm)
        else:
            contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos, task_ind)
        loss = rec_loss + contrastive_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        recon_loss += rec_loss.detach().item()
        contra_loss += contrastive_loss.detach().item()
        if counter % 500 == 0:
            print(f"[{counter}/{len_dataloader}]")
        counter += 1
        
    return recon_loss / len(data_loader), contra_loss / len(data_loader)

def train_loop_classification(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, classification_head, opt, loss_fn, data_loader):
    encoder.train()
    prediction_head_state.train()
    prediction_head_action.train()
    prediction_head_reward.train()
    classification_head.train()
    recon_loss = 0
    class_loss = 0
    counter = 0
    len_dataloader = len(data_loader)

    state_indices = torch.arange(0, sequence_len*3, 3) 
    action_indices = torch.arange(1, (sequence_len-1)*3, 3)
    reward_indices = torch.arange(2, (sequence_len-1)*3, 3)
    
    for batch in data_loader:
        # select feature for the input and output
        states, actions, _, rewards, task_obs, _ = batch
        states = states.to(device)
        actions = actions[:, :-1].to(device)
        rewards = rewards[:, :-1].to(device)

        task_indices = torch.tensor([torch.where(unique_labels == label)[0].item() for label in task_obs]).to(device)

        state_inp, state_inp_mask, action_inp, action_inp_mask, reward_inp, reward_inp_mask = mask_tokens(states, actions, rewards)
        
        # forward pass trough the network
        encoding = encoder(state_inp, action_inp, reward_inp)
        encoding_state, encoding_action, encoding_reward, encoding_cls_token = encoding[state_indices], encoding[action_indices], encoding[reward_indices], encoding[-1]
        pred_state = prediction_head_state(encoding_state).permute(1, 0, 2)
        pred_action = prediction_head_action(encoding_action).permute(1, 0, 2)
        pred_reward = prediction_head_reward(encoding_reward).permute(1, 0, 2)
        class_label = classification_head(encoding_cls_token)

        pred_state, state_tar = pred_state[state_inp_mask], states[state_inp_mask]
        pred_action, action_tar = pred_action[action_inp_mask], actions[action_inp_mask]
        pred_reward, reward_tar = pred_reward[reward_inp_mask], rewards[reward_inp_mask]

        # calculate the state-/action-/reward-loss independently
        rec_loss = loss_fn(pred_state, state_tar) + loss_fn(pred_action, action_tar) + loss_fn(pred_reward, reward_tar)
        classification_loss = F.cross_entropy(class_label, task_indices.to(device))
        loss = rec_loss + classification_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        recon_loss += rec_loss.detach().item()
        class_loss += classification_loss.detach().item()
        if counter % 500 == 0:
            print(f"[{counter}/{len_dataloader}]")
        counter += 1
        
    return recon_loss / len(data_loader), class_loss / len(data_loader)

def validation_loop(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, prediction_head_cls, loss_fn, contrastive_loss_fn, data_loader):
    encoder.eval()
    prediction_head_state.eval()
    prediction_head_action.eval()
    prediction_head_reward.eval()
    prediction_head_cls.eval()
    recon_loss = 0
    contra_loss = 0

    state_indices = torch.arange(0, sequence_len*3, 3) 
    action_indices = torch.arange(1, (sequence_len-1)*3, 3)
    reward_indices = torch.arange(2, (sequence_len-1)*3, 3)
    
    with torch.no_grad():
        for batch in data_loader:
            # select feature for the input and output
            states, actions, _, rewards, task_ind, task_arm = batch
            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)
            task_ind = task_ind.to(device)
            task_arm = task_arm.to(device)

            state_inp, state_inp_mask, action_inp, action_inp_mask, reward_inp, reward_inp_mask = mask_tokens(states, actions, rewards)
            
            # forward pass trough the network
            encoding = encoder(state_inp, action_inp, reward_inp)
            encoding_state, encoding_action, encoding_reward, encoding_cls_token = encoding[state_indices], encoding[action_indices], encoding[reward_indices], encoding[-1]
            pred_state = prediction_head_state(encoding_state).permute(1, 0, 2)
            pred_action = prediction_head_action(encoding_action).permute(1, 0, 2)
            pred_reward = prediction_head_reward(encoding_reward).permute(1, 0, 2)
            cls_token = prediction_head_cls(encoding_cls_token)

            pred_state, state_tar = pred_state[state_inp_mask], states[state_inp_mask]
            pred_action, action_tar = pred_action[action_inp_mask], actions[action_inp_mask]
            pred_reward, reward_tar = pred_reward[reward_inp_mask], rewards[reward_inp_mask]

            # positive samples
            state_inp_pos, state_inp_mask_pos, action_inp_pos, action_inp_mask_pos, reward_inp_pos, reward_inp_mask_pos = mask_tokens(states, actions, rewards)
            
            # forward pass trough the network
            encoding_pos = encoder(state_inp_pos, action_inp_pos, reward_inp_pos)
            encoding_state_pos, encoding_action_pos, encoding_reward_pos, encoding_cls_token_pos = encoding_pos[state_indices], encoding_pos[action_indices], encoding_pos[reward_indices], encoding_pos[-1]
            pred_state_pos = prediction_head_state(encoding_state_pos).permute(1, 0, 2)
            pred_action_pos = prediction_head_action(encoding_action_pos).permute(1, 0, 2)
            pred_reward_pos = prediction_head_reward(encoding_reward_pos).permute(1, 0, 2)
            cls_token_pos = prediction_head_cls(encoding_cls_token_pos)

            pred_state_pos, state_tar_pos = pred_state_pos[state_inp_mask_pos], states[state_inp_mask_pos]
            pred_action_pos, action_tar_pos = pred_action_pos[action_inp_mask_pos], actions[action_inp_mask_pos]
            pred_reward_pos, reward_tar_pos = pred_reward_pos[reward_inp_mask_pos], rewards[reward_inp_mask_pos]

            # calculate the state-/action-/reward-loss independently
            rec_loss = loss_fn(pred_state, state_tar) + loss_fn(pred_action, action_tar) + loss_fn(pred_reward, reward_tar) + \
                     loss_fn(pred_state_pos, state_tar_pos) + loss_fn(pred_action_pos, action_tar_pos) + loss_fn(pred_reward_pos, reward_tar_pos)
            if not use_own_contrastive_loss:
                contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos)
            elif use_task_arm:
                contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(cls_token, cls_token_pos, task_arm)
            else:
                contrastive_loss = contrastive_loss_weight * contrastive_loss_fn(query=cls_token, positive_key=cls_token_pos, task_ind=task_ind)
            loss = rec_loss + contrastive_loss

            recon_loss += rec_loss.detach().item()
            contra_loss += contrastive_loss.detach().item()
        
    return recon_loss / len(data_loader), contra_loss / len(data_loader)

def predict(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, data_loader, sample_count=5):
    encoder.eval()
    prediction_head_state.eval()
    prediction_head_action.eval()
    prediction_head_reward.eval()
    total_loss = 0
    data_loader = iter(data_loader)

    state_indices = torch.arange(0, sequence_len*3, 3) 
    action_indices = torch.arange(1, (sequence_len-1)*3, 3)
    reward_indices = torch.arange(2, (sequence_len-1)*3, 3)

    with torch.no_grad():
        for i in range(sample_count):
            sample = next(data_loader)
            states, actions, _, rewards, _, _ = sample
            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)

            encoding = encoder(states, actions, rewards)
            encoding_state, encoding_action, encoding_reward = encoding[state_indices], encoding[action_indices], encoding[reward_indices]
            pred_state = prediction_head_state(encoding_state).permute(1, 0, 2)
            pred_action = prediction_head_action(encoding_action).permute(1, 0, 2)
            pred_reward = prediction_head_reward(encoding_reward).permute(1, 0, 2)

            state_loss = loss_fn(pred_state, states)
            action_loss = loss_fn(pred_action, actions)
            reward_loss = loss_fn(pred_reward, rewards)
            loss = state_loss + action_loss + reward_loss

            # NO contrastive loss since we only look at single sample

            total_loss += loss.detach().item()

            print(f"States predicition: \n{pred_state}")
            print(f"States target: \n{states}")
            print(f"State loss: {state_loss}")
            print(f"Action predicition: \n{pred_action}")
            print(f"Action target: \n{actions}")
            print(f"Action loss: {action_loss}")
            print(f"Reward predicition: \n{pred_reward}")
            print(f"Reward target: \n{rewards}")
            print(f"Reward loss: {reward_loss}")
            print("-"*25 + f" Sample {i} "+"-"*25)

    return total_loss / sample_count

def predict_last_state(encoder, prediction_head_state, data_loader):
    encoder.eval()
    prediction_head_state.eval()
    total_loss = 0
    data_loader = iter(data_loader)

    with torch.no_grad():
        for batch in data_loader:
            states, actions, _, rewards, _, _ = batch
            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)

            state_inp, _ = mask_last_state(states)

            encoding = encoder(state_inp, actions, rewards)
            encoding_state = encoding[-1]
            pred_state = prediction_head_state(encoding_state)

            state_loss = loss_fn(pred_state, states[:, -1])
            loss = state_loss

            # NO contrastive loss since we only look at single sample

            total_loss += loss.detach().item()

    return total_loss / len(data_loader)

def embedding_valdation(encoder, prediction_head_cls, samples_num, data_loader, save_path):
    encoder.eval()
    prediction_head_cls.eval()
    
    counter = 0
    z_arr, t_arr, task_arr, states_arr, action_arr, reward_arr, task_arm_arr, summed_reward_arr = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in data_loader:
            states, actions, _, rewards, task_obs, task_arm = batch

            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)

            states_arr.append(states)
            action_arr.append(actions)
            reward_arr.append(rewards)

            counter += states.shape[0]

            task_arr.append(task_obs)
            task_arm_arr.append(task_arm)

            summed_reward_arr.append(torch.sum(rewards, dim=1))
           
            encoding = encoder(states, actions, rewards)
            encoding = encoding.permute(1, 0, 2)

            state_embedding = encoding[:, -2]

            if cluster_state:
                trajectory_embedding = encoding[:, -2]
            else:
                trajectory_embedding = prediction_head_cls(encoding[:, -1])
            #trajectory_embedding = encoding[:, -1]

            t_arr.append(trajectory_embedding)
            z_arr.append(state_embedding)

            if counter>samples_num:
                break
    task = torch.cat(task_arr).cpu().detach().numpy()
    task_arm = torch.cat(task_arm_arr).cpu().detach().numpy()
    rewards = torch.cat(summed_reward_arr).cpu().detach().numpy()
    z = torch.cat(z_arr).cpu().detach().numpy()
    t = torch.cat(t_arr).cpu().detach().numpy()
    obs_state = torch.cat(states_arr).cpu().detach().numpy()
    obs_actions = torch.cat(action_arr).cpu().detach().numpy()
    obs_rewards = torch.cat(reward_arr).cpu().detach().numpy()
    print(f"Saving embeddings {save_path}...")
    torch.save({"task": task, "tra_emb":t, "state_emb":z, "rewards": rewards, "task_arm": task_arm, "states": obs_state, "actions": obs_actions, "rew": obs_rewards}, save_path)

    print("saving cluster assignment")
    manage_latent_representation(torch.cat(t_arr), torch.cat(task_arr), 'end')

def trajectory_embedding_valdation(encoder, prediction_head_cls, samples_num, dataset, save_path):
    encoder.eval()
    prediction_head_cls.eval()

    avaible_length = loaded_dataset.avaible_length    
    counter = 0
    z_arr, t_arr, task_arr, states_arr, action_arr, reward_arr, summed_reward_arr, task_arm_arr, time_step_arr = [], [], [], [], [], [], [], [], []
    unique_env_idx = np.unique(dataset.task_obs)
    with torch.no_grad():
        while counter < samples_num:
            sample_per_task = []
            for idx in unique_env_idx:
                mask = dataset.task_obs == idx
                possible_values = np.arange(len(dataset.task_obs))[mask]
                samples = np.random.choice(possible_values)
                samples = np.arange(avaible_length) + samples * dataset.avaible_length
                sample_per_task.append(samples)
            sample_per_task = np.concatenate(sample_per_task)
            batch = [dataset[idx] for idx in sample_per_task]

            # Now you can convert your batch into tensors, or further process it as needed
            states = torch.tensor(np.stack([item[0] for item in batch]))
            actions= torch.tensor(np.stack([item[1] for item in batch]))
            rewards = torch.tensor(np.stack([item[3] for item in batch]))
            task_obs = torch.tensor(np.stack([item[4] for item in batch]))
            task_arm = torch.tensor(np.stack([item[5] for item in batch]))

            counter += avaible_length * len(unique_env_idx)

            states = states.to(device)
            actions = actions[:, :-1].to(device)
            rewards = rewards[:, :-1].to(device)

            states_arr.append(states)
            action_arr.append(actions)
            reward_arr.append(rewards)
            time_step_arr.append(np.tile(np.arange(avaible_length), len(unique_env_idx)))

            task_arr.append(task_obs)
            task_arm_arr.append(task_arm)

            summed_reward_arr.append(torch.sum(rewards, dim=1))
           
            encoding = encoder(states, actions, rewards)
            encoding = encoding.permute(1, 0, 2)

            state_embedding = encoding[:, -2]

            if cluster_state:
                trajectory_embedding = encoding[:, -2]
            else:
                trajectory_embedding = prediction_head_cls(encoding[:, -1])

            t_arr.append(trajectory_embedding)
            z_arr.append(state_embedding)

    task = torch.cat(task_arr).cpu().detach().numpy()
    task_arm = torch.cat(task_arm_arr).cpu().detach().numpy()
    rewards = torch.cat(summed_reward_arr).cpu().detach().numpy()
    z = torch.cat(z_arr).cpu().detach().numpy()
    t = torch.cat(t_arr).cpu().detach().numpy()
    obs_state = torch.cat(states_arr).cpu().detach().numpy()
    obs_actions = torch.cat(action_arr).cpu().detach().numpy()
    obs_rewards = torch.cat(reward_arr).cpu().detach().numpy()
    time_steps = np.concatenate(time_step_arr)
    print("Saving embeddings...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"task": task, "tra_emb": t, "state_emb": z, "rewards": rewards, "task_arm": task_arm, "states": obs_state, "actions": obs_actions, "rew": obs_rewards, "timesteps": time_steps}, save_path)

    print("saving cluster assignment")
    manage_latent_representation(torch.cat(t_arr), torch.cat(task_arr), 'end')

# ------ bnpy ----------
def bnpy_cluster(encoder, prediction_head_cls, data_loader):
    encoder.eval()
    prediction_head_cls.eval()

    if not os.path.exists(os.path.join(bnpy_save_dir, 'birth_debug')):
        os.makedirs(os.path.join(bnpy_save_dir, 'birth_debug'))
    if not os.path.exists(os.path.join(bnpy_save_dir, 'data')):
        os.makedirs(os.path.join(bnpy_save_dir, 'data'))
    
    for i in range(bnpy_epochs):
        counter = 0
        z_arr, task_arr = [], []
        with torch.no_grad():
            for batch in data_loader:
                states, actions, _, rewards, task_obs, _ = batch

                states = states.to(device)
                actions = actions[:, :-1].to(device)
                rewards = rewards[:, :-1].to(device)

                counter += states.shape[0]

                task_arr.append(task_obs)
            
                encoding = encoder(states, actions, rewards)
                encoding = encoding.permute(1, 0, 2)


                if cluster_state:
                    trajectory_embedding = encoding[:, -2]
                else:
                    trajectory_embedding = prediction_head_cls(encoding[:, -1])

                z_arr.append(trajectory_embedding)

                if counter>sample_per_iter:
                    break
        task = torch.cat(task_arr)
        z = torch.cat(z_arr)

        print(f"--- Fitting bnpy at step [{i}] ---")
        out_path = os.path.join(bnpy_save_dir, str(next(iterator)))
        fit(z, out_path)

    manage_latent_representation(z, task, 'end')
    print("Saving model ...")
    save_bnpy_model(bnpy_save_dir, out_path + '/1')

def fit(z, out_path):
        '''
        fit the model, input z should be torch.tensor format
        '''
        global bnpy_model
        global info_dict

        z = XData(z.detach().cpu().numpy())
        if not bnpy_model:
            print("=== Initialing DPMM model ===")
            bnpy_model, info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB',
                                                  output_path=out_path,
                                                  #output_path=bnpy_save_dir,
                                                  initname='randexamples',
                                                  K=1, 
                                                  gamma0=gamma0,
                                                  sF=sF, 
                                                  ECovMat='eye',
                                                  moves='birth,merge', 
                                                  nBatch=4, 
                                                  nLap=num_lap,
                                                  **dict(
                                                      sum(map(list, [birth_kwargs.items(),
                                                                     merge_kwargs.items()]), []))
                                                  )
            print("=== DPMM model initialized ===")
        else:
            bnpy_model, info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB',
                                                  output_path=out_path,
                                                  #output_path=bnpy_save_dir,
                                                  initname=info_dict['task_output_path'],
                                                  K=info_dict['K_history'][-1], 
                                                  gamma0=gamma0,
                                                  sF=sF, 
                                                  ECovMat='eye',
                                                  moves='birth,merge', 
                                                  nBatch=4, 
                                                  nLap=num_lap,
                                                  **dict(
                                                      sum(map(list, [birth_kwargs.items(),
                                                                     merge_kwargs.items()]), []))
                                                  )
        calc_cluster_component_params()

def calc_cluster_component_params():
    global comp_mu
    global comp_var
    comp_mu = [torch.Tensor(bnpy_model.obsModel.get_mean_for_comp(i))
                    for i in np.arange(0, bnpy_model.obsModel.K)]
    comp_var = [torch.Tensor(np.sum(bnpy_model.obsModel.get_covar_mat_for_comp(i), axis=0))
                        for i in np.arange(0, bnpy_model.obsModel.K)]
    
def cluster_assignments(z):
    z = XData(z.detach().cpu().numpy())
    LP = bnpy_model.calc_local_params(z)
    # Here, resp is a 2D array of size N x K.
    # Each entry resp[n, k] gives the probability
    # that data atom n is assigned to cluster k under
    # the posterior.
    resp = LP['resp']
    # To convert to hard assignments
    # Here, Z is a 1D array of size N, where entry Z[n] is an integer in the set {0, 1, 2, … K-1, K}.
    Z = resp.argmax(axis=1)
    return resp, Z

def cluster_assignments_with_comp_para(z):
    z = XData(z.detach().cpu().numpy())
    LP = bnpy_model.calc_local_params(z)
    Z = LP['resp'].argmax(axis=1)

    comp_mu = [torch.Tensor(bnpy_model.obsModel.get_mean_for_comp(i))
                    for i in np.arange(0, bnpy_model.obsModel.K)]
    
    m = torch.stack([comp_mu[x] for x in Z])
    return m, Z

def manage_latent_representation(z, env_idx, prefix:str=''):
    '''
    calculate the latent z and their corresponding clusters, save the values for later evaluation
    saved value:
        z: latent encoding from VAE Encoder
        env_idx: corresponding env idx of each z (true label)
        env_name: corresponding env name of each z
        cluster_label: corresponding cluster No. that z should be assigned to
        cluster_param: corresponding cluster parameters (mu & var of diagGauss) of each z
    '''
    
    comps_mu_list = []
    comps_var_list = []
    # cluster label
    _, cluster_label = cluster_assignments(z)
    # get clusters param
    comp_mu = [bnpy_model.obsModel.get_mean_for_comp(i) for i in np.arange(0, bnpy_model.obsModel.K)]
    comp_var = [np.sum(bnpy_model.obsModel.get_covar_mat_for_comp(i), axis=0) for i in np.arange(0, bnpy_model.obsModel.K)]
    for i in cluster_label:
        comps_mu_list.append(comp_mu[i])
        comps_var_list.append(comp_var[i])
    # summarize data into dict
    data = dict(
        z=z.detach().cpu().numpy(),
        env_idx=env_idx.detach().cpu().numpy(),
        cluster_label=cluster_label,
        cluster_mu=comps_mu_list,
        cluster_var=comps_var_list,
    )
    # save the file
    np.savez(bnpy_save_dir+'/data'+'/latent_samples_{}.npz'.format(prefix), **data)

def load_bnpy_model(bnpy_load_path):
    global bnpy_model
    global info_dict
    if os.path.exists(bnpy_load_path) and os.path.isdir(bnpy_load_path):
        bnpy_model = bnpy.load_model_at_lap(bnpy_load_path, 100)[0]
        info_dict = np.load(bnpy_load_path+'/info_dict.npy', allow_pickle=True).item()
    else:
        print(f"bnpy model at {bnpy_load_path} not found!!!")

def save_info_dict(save_path, new_output_path=None):
    if new_output_path:
        info = dict(
            task_output_path = new_output_path,
            K_history=[info_dict['K_history'][-1]],
        )
    else:
        info = dict(
            task_output_path = info_dict['task_output_path'],
            K_history=[info_dict['K_history'][-1]],
        )
    np.save(save_path+'/info_dict.npy', info)

def save_bnpy_model(save_path, current_dict):
    if os.path.exists(current_dict) and os.path.isdir(current_dict):
        if os.path.exists(save_path + '/save'):
            shutil.rmtree(save_path + '/save')
        print(f"Folder at {current_dict} already exists")
    shutil.copytree(current_dict, save_path + '/save')
    save_info_dict(save_path + '/save', save_path + '/save')
    save_comps_parameters(save_path + '/data')

def save_comps_parameters(save_path):
    '''
    save the model param in form of *npy
    '''
    comp_mu = [bnpy_model.obsModel.get_mean_for_comp(i)
                    for i in np.arange(0, bnpy_model.obsModel.K)]
    comp_var = [np.sum(bnpy_model.obsModel.get_covar_mat_for_comp(i), axis=0) # save diag value in a 1-dim array
                        for i in np.arange(0, bnpy_model.obsModel.K)]
    data = dict(
        comp_mu=comp_mu,
        comp_var=comp_var
    )
    np.save(save_path+'/comp_params.npy', data)
# ------ bnpy ----------


def mask_tokens(input_state, input_action, input_reward, mask_prob=0.15, unchanged_prob=0.10, random_prob=0.10):
    state_feature = input_state.shape[-1]
    action_feature = input_action.shape[-1]

    masked_state_tensor = input_state.clone().to(device)
    masked_action_tensor = input_action.clone().to(device)
    masked_reward_tensor = input_reward.clone().to(device)
    
    mask_state = torch.rand((masked_state_tensor.shape[:2]), device=device) < mask_prob
    mask_action = torch.rand((masked_action_tensor.shape[:2]), device=device) < mask_prob
    mask_reward = torch.rand((masked_reward_tensor.shape[:2]), device=device) < mask_prob

    ensure_masking = torch.zeros(mask_state.shape[1], dtype=torch.bool).to(device)
    ensure_masking[-1] = True
    mask_state[(mask_state == False).all(dim=1)] = ensure_masking
    ensure_masking = torch.zeros(mask_action.shape[1], dtype=torch.bool).to(device)
    ensure_masking[-1] = True
    mask_action[(mask_action == False).all(dim=1)] = ensure_masking
    mask_reward[(mask_reward == False).all(dim=1)] = ensure_masking

    probability_sample_state = torch.rand(masked_state_tensor.shape[:2], device=device)
    probability_sample_action = torch.rand(masked_action_tensor.shape[:2], device=device)
    probability_sample_reward = torch.rand(masked_reward_tensor.shape[:2], device=device)
    random_mask_state = probability_sample_state < random_prob
    unchanged_mask_state = probability_sample_state > (random_prob+unchanged_prob)
    random_mask_action = probability_sample_action < random_prob
    unchanged_mask_action = probability_sample_action > (random_prob+unchanged_prob)
    random_mask_reward = probability_sample_reward < random_prob
    unchanged_mask_reward = probability_sample_reward > (random_prob+unchanged_prob)
    
    masked_state_tensor[unchanged_mask_state & mask_state] = torch.zeros(state_feature, device=device)
    masked_state_tensor[random_mask_state & mask_state] = torch.rand(state_feature, device=device)
    masked_action_tensor[unchanged_mask_action & mask_action] = torch.zeros(action_feature, device=device)
    masked_action_tensor[random_mask_action & mask_action] = torch.rand(action_feature, device=device)
    masked_reward_tensor[unchanged_mask_reward & mask_reward] = torch.zeros(1, device=device)
    masked_reward_tensor[random_mask_reward & mask_reward] = torch.rand(1, device=device)

    return masked_state_tensor, mask_state, masked_action_tensor, mask_action, masked_reward_tensor, mask_reward

def mask_last_state(input_state):
    state_feature = input_state.shape[-1]

    masked_state_tensor = input_state.clone().to(device)
    
    mask_state = torch.zeros(masked_state_tensor.shape[:2], dtype=torch.bool)
    mask_state[:, -1] = True
    
    masked_state_tensor[mask_state] = torch.zeros(state_feature, device=device)

    return masked_state_tensor, mask_state

def euclidean_distance(vec1, vec2):
    # Calculate the Euclidean distance
    distance = torch.sqrt(torch.sum((vec1 - vec2) ** 2))
    
    return distance.item()

def save_model(model, prediciton_head_state, prediciton_head_action, prediciton_head_reward, prediction_head_cls, model_path):
    """Saves the model and decoder state dictionaries to the specified path."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'prediciton_head_state': prediciton_head_state.state_dict(),
        'prediciton_head_action': prediciton_head_action.state_dict(),
        'prediciton_head_reward': prediciton_head_reward.state_dict(),
        'prediction_head_cls': prediction_head_cls.state_dict(),
    }, model_path)
    print(f"Model and decoder saved to {model_path}")

def load_model(model, prediciton_head_state, prediciton_head_action, prediciton_head_reward, prediction_head_cls, model_path):
    """Loads the model and decoder state dictionaries from the specified path."""
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        prediciton_head_state.load_state_dict(checkpoint['prediciton_head_state'])
        prediciton_head_action.load_state_dict(checkpoint['prediciton_head_action'])
        prediciton_head_reward.load_state_dict(checkpoint['prediciton_head_reward'])
        prediction_head_cls.load_state_dict(checkpoint['prediction_head_cls'])
        print(f"Model and decoder loaded from {model_path}")
    else:
        print(f"No checkpoint found at {model_path}. Initializing model from scratch.")

def set_seed(seed):
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

########################### Hyperparameter ###########################

state_dim = 21
action_dim = 4
cls_dim = 6

d_model = 32 #20 32 64 | Model dimension
nhead = 8 # 4 8 | Number of attention heads 
num_layers = 3  # Number of transformer encoder layers 
dim_feedforward = 128 #128 #256
dropout = 0.1 
mask_prob = 0.2
mask_unchanged = 0.15
random_mask_prob = 0.1

project_root = os.environ.get("PROJECT_ROOT")
bnpy_save_dir = project_root + '/Transformer_RNN/bnpy_save'

gamma0=5.0
num_lap=20
sF=1. #0.1
bnpy_model = None
info_dict = None
iterator = cycle(range(2))
comp_mu = None
comp_var = None
sample_per_iter = 10_000
birth_kwargs= {
    'b_startLap': 1, 
    'b_stopLap': 2, 
    'b_Kfresh': 10, 
    'b_minNumAtomsForNewComp': 10.0, 
    'b_minNumAtomsForTargetComp': 10.0, 
    'b_minNumAtomsForRetainComp': 10.0, 
    'b_minPercChangeInNumAtomsToReactivate': 0.03, 
    'b_debugWriteHTML': 0
    }
merge_kwargs={
    'm_startLap': 2, 
    'm_maxNumPairsContainingComp': 50, 
    'm_nLapToReactivate': 1, 
    'm_pair_ranking_procedure': 'obsmodel_elbo', 
    'm_pair_ranking_direction': 'descending'
    }

batch_size = 128
learning_rate = 0.0002
epochs = 4
sequence_len = 20
predict_samples_num = 0
bnpy_epochs = 1

contrastive_temp = 0.15 #0.25
contrastive_loss_weight = 0.25
use_own_contrastive_loss = True
use_task_arm = False
cluster_state = False

should_continue = True
should_safe = True
evaluate_embedding = True
should_continue_bnpy = False
train_bnpy = True
should_predict_last_state = True

device = "cuda" if torch.cuda.is_available() else "cpu"
seed=2

embedding_num = 3_000 #20_000
model_path = 'Transformer_RNN/checkpoints/representation_cls_transformer_checkpoint.pth'
dataset_path = 'Transformer_RNN/decision_tf_dataset/train/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/recorded_envs/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/recorded_faucet/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/eval_embodiments/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/kuka_saywer/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/expert_10/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/distill+kuka+saywer/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/distill+saywer/'
#dataset_path = 'Transformer_RNN/decision_tf_dataset/expert_30/'

val_dataset_path = 'Transformer_RNN/decision_tf_dataset/validation/'
#val_dataset_path = 'Transformer_RNN/decision_tf_dataset/recorded_faucet/'
#val_dataset_path = 'Transformer_RNN/decision_tf_dataset/eval_embodiments/'

embeddings_path = 'Transformer_RNN/embedding_log/emb.pth'
log_path = 'Transformer_RNN/tensorboard_log'

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
    # Set initial seed for reproducibility
    set_seed(seed)

    # Load the dataset
    loaded_dataset = TFDataset.load(dataset_path, sequence_length=sequence_len) # here one more
    loaded_val_dataset = TFDataset.load(val_dataset_path, sequence_length=sequence_len)
    total_size = len(loaded_val_dataset)
    val_size = int(0.3 * total_size)
    not_used_size = 0 #int(0.5 * total_size)
    test_size =  total_size - val_size - not_used_size
    unique_labels = torch.tensor(loaded_dataset.unique_task_obs) 

    # create Dataloader
    val_dataset, test_dataset, _ = random_split(loaded_val_dataset, [val_size, test_size, not_used_size])
    train_loader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    single_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create the Encoder and simple MLP
    encoder = EncoderOnlyTransformerModel(state_dim=state_dim,
                                          action_dim=action_dim,
                                          d_model=d_model,
                                          nhead=nhead,
                                          dim_feedforward=dim_feedforward,
                                          nlayers=num_layers, 
                                          sequence_len=sequence_len,
                                          device=device,
                                          dropout=dropout).to(device)
    prediction_head_state = PredictionHead(d_model, state_dim).to(device)
    prediction_head_action = PredictionHead(d_model, action_dim).to(device)
    prediction_head_reward = PredictionHead(d_model, 1).to(device)
    prediction_head_cls = PredictionHead(d_model, cls_dim).to(device)
    classification_head = ClassificationHead(d_model, len(unique_labels)).to(device)

    if should_continue:
        load_model(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, prediction_head_cls, model_path)
    if should_continue_bnpy:
        load_bnpy_model(bnpy_save_dir + '/save')

    # Creating optimizer and loss function
    opt = torch.optim.Adam(list(encoder.parameters()) + list(prediction_head_state.parameters()) + \
                           list(prediction_head_action.parameters()) + list(prediction_head_reward.parameters()) + \
                           list(prediction_head_cls.parameters()) + list(classification_head.parameters()), lr=learning_rate)
    loss_fn = nn.MSELoss(reduction='mean')
    contrastive_loss = InfoNCE(temperature=contrastive_temp)

    # Initialize the SummaryWriter
    writer = SummaryWriter(log_dir=log_path)

    # Training loop
    print("Training and validating model")
    start_time = time.time()
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        rec_loss, con_loss = train_loop(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, prediction_head_cls, opt, loss_fn, contrastive_loss,  train_loader)
        #rec_loss, con_loss = train_loop_classification(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, classification_head, opt, loss_fn, train_loader)
        writer.add_scalar('Loss/train_rec', rec_loss, epoch)
        writer.add_scalar('Loss/train_cont', con_loss, epoch)
        
        val_rec_loss, val_con_loss = validation_loop(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, prediction_head_cls, loss_fn, contrastive_loss, val_loader)
        writer.add_scalar('Loss/validation_rec', val_rec_loss, epoch)
        writer.add_scalar('Loss/validation_cont', val_con_loss, epoch)
        
        print(f"Training reconstruction loss: {rec_loss:.4f} | Training contrastive loss: {con_loss:.4f} | Time: {time.time() - start_time}")
        print(f"Validation reconstructionloss: {val_rec_loss:.4f} | Validation contrastive loss: {val_con_loss:.4f} \n")

        if should_safe:
            save_model(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, prediction_head_cls, model_path)

        start_time = time.time()

    if predict_samples_num > 0:
        predict(encoder, prediction_head_state, prediction_head_action, prediction_head_reward, prediction_head_cls, single_test_loader, predict_samples_num)

    if should_predict_last_state:
        last_state_loss = predict_last_state(encoder, prediction_head_state, test_loader)
        writer.add_scalar('Loss/last_state', last_state_loss, 0)
        print(f"Loss on prediciting the last state on the testset: {last_state_loss}")

    if train_bnpy:
        bnpy_cluster(encoder, prediction_head_cls, train_loader)

    if evaluate_embedding:
        trajectory_embedding_valdation(encoder, prediction_head_cls, embedding_num, loaded_dataset, embeddings_path)
        #embedding_valdation(encoder, prediction_head_cls, embedding_num, train_loader, embeddings_path)
        #embedding_valdation(encoder, prediction_head_cls, embedding_num, val_loader, embeddings_path)

    # Close the SummaryWriter when done
    writer.close()
