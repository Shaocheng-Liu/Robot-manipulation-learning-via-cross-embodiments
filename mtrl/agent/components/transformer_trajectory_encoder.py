import torch
import torch.nn as nn
import numpy as np
import math
import os

import bnpy
from bnpy.data.XData import XData


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

    def __init__(self, ntoken, d_model, nhead, dim_feedforward, nlayers, compression_dim, sequence_len, model_path, with_cls, dropout=0.25):
        if with_cls:
            sequence_len += 1 # +1 for CLS token

        super(EncoderOnlyTransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, sequence_len)

        self.input_emb = nn.Linear(ntoken, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.compression_dim = compression_dim
        if self.compression_dim > 0:
            self.output_emb = nn.Linear(d_model, compression_dim)
        self.d_model = d_model

        if model_path is not None:
            self.load_model(model_path)
        else:
            self.init_weights()

    def init_weights(self): #TODO
        initrange = 0.1
        nn.init.zeros_(self.input_emb.bias)
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        if self.compression_dim > 0:
            nn.init.zeros_(self.output_emb.bias)
            nn.init.uniform_(self.output_emb.weight, -initrange, initrange)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.input_emb(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        if self.compression_dim > 0:
            output = self.output_emb(output)
        return output

    def load_model(self, model_path):
        """Loads the model and decoder state dictionaries from the specified path."""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model and decoder loaded from {model_path}")
        else:
            print(f"No checkpoint found at {model_path}. Initializing model from scratch.")

class RepresentationEncoderTransformer(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, state_dim, action_dim, d_model, nhead, dim_feedforward, nlayers, sequence_len, model_path, dropout=0.1):
        super(RepresentationEncoderTransformer, self).__init__()
        self.pos_encoder = TrainablePositionalEncoding(d_model, dropout, sequence_len)

        self.state_emb = nn.Linear(state_dim, d_model)
        self.action_emb = nn.Linear(action_dim, d_model)
        self.reward_emb = nn.Linear(1, d_model)

        self.state_norm = nn.LayerNorm(d_model)
        self.action_norm = nn.LayerNorm(d_model)
        self.reward_norm = nn.LayerNorm(d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        self.special_token = nn.Parameter(torch.full((1, 1, d_model), -2.))

        if model_path is not None:
            self.load_model(model_path)
        else:
            print("No pretrained model found for RepresentationEncoderTransformer")

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def forward(self, states_src, actions_src, rewards_src, src_key_padding_mask=None):
        batch_size, seq_len = actions_src.shape[0], actions_src.shape[1]

        state_token_src = self.state_norm(self.state_emb(states_src))
        action_token_src = self.action_norm(self.action_emb(actions_src))
        reward_token_src = self.reward_norm(self.reward_emb(rewards_src))

        state_token_src = self.pos_encoder(state_token_src.permute(1, 0, 2))
        reward_token_src = self.pos_encoder(reward_token_src.permute(1, 0, 2))
        action_token_src = self.pos_encoder(action_token_src.permute(1, 0, 2))

        special_token = self.special_token.expand(-1, batch_size, -1)

        if seq_len > 0:
            src = torch.stack((action_token_src, reward_token_src, state_token_src[1:]), dim=0).transpose(0,1).reshape(3*seq_len, -1, self.d_model) # shape: 30, batchsize, hidden size
            src = torch.cat((state_token_src[0][None], src, special_token), dim=0)
        else:
            src = torch.cat((state_token_src[0][None], special_token), dim=0)
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output
    
    # TODO
    # def save_model(model, prediciton_head_state, prediciton_head_action, prediciton_head_reward, model_path):
    #     """Saves the model and decoder state dictionaries to the specified path."""
    #     torch.save({
    #         'model_state_dict': model.state_dict(),
    #         'prediciton_head_state': prediciton_head_state.state_dict(),
    #         'prediciton_head_action': prediciton_head_action.state_dict(),
    #         'prediciton_head_reward': prediciton_head_reward.state_dict(),
    #     }, model_path)
    #     print(f"Model and decoder saved to {model_path}")

    def load_model(self, model_path):
        """Loads the model and decoder state dictionaries from the specified path."""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model and decoder loaded from {model_path}")
        else:
            print(f"No checkpoint found at {model_path}. Initializing model from scratch.")


class ClsPredictionHead(nn.Module):
    def __init__(self, encoding_dim, latent_dim, model_path):
        super(ClsPredictionHead, self).__init__()
        self.linear = nn.Linear(encoding_dim, latent_dim)

        if model_path is not None:
            self.load_model(model_path)
        else:
            print("No pretrained model found for ClsPredictionHead")

    def forward(self, encoding):
        return self.linear(encoding)
    
    def load_model(self, model_path):
        """Loads the model and decoder state dictionaries from the specified path."""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.load_state_dict(checkpoint['prediction_head_cls'])
            print(f"Model and decoder loaded from {model_path}")
        else:
            print(f"No checkpoint found at {model_path}. Initializing model from scratch.")
    
class Bnpy_model():
    def __init__(self, bnpy_load_path, device):
        self.bnpy_model= None
        #self.info_dict = None
        self.device = device
        if os.path.exists(bnpy_load_path) and os.path.isdir(bnpy_load_path):
            self.bnpy_model = bnpy.load_model_at_lap(bnpy_load_path, 100)[0]
            #self.info_dict = np.load(bnpy_load_path+'/info_dict.npy', allow_pickle=True).item()
        else:
            print(f"bnpy model at {bnpy_load_path} not found!!!")

    def cluster(self, z):
        z = XData(z.detach().cpu().numpy())
        LP = self.bnpy_model.calc_local_params(z)
        Z = LP['resp'].argmax(axis=1)

        comp_mu = [torch.Tensor(self.bnpy_model.obsModel.get_mean_for_comp(i))
                        for i in np.arange(0, self.bnpy_model.obsModel.K)]
        
        m = torch.stack([comp_mu[x] for x in Z])
        m = m.to(self.device)
        return m, Z
