import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
from dataset_tf import TFDataset
import os
from torch.utils.tensorboard import SummaryWriter

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n
    
class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(GRUEncoder, self).__init__()
        
        # GRU layer
        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True, 
                          bidirectional=bidirectional)
        
    def forward(self, x):        
        # Pass through GRU layer
        _, out = self.gru(x)
        return out

class Predictor(nn.Module):
    def __init__(self, encoding_dim, output_dim):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(encoding_dim, output_dim)

    def forward(self, encoding):
        return self.linear(encoding)

def save_model(encoder, decoder, model_path):
    """Saves the model and decoder state dictionaries to the specified path."""
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, model_path)
    print(f"Model and decoder saved to {model_path}")

def load_model(encoder, decoder, model_path):
    """Loads the model and decoder state dictionaries from the specified path."""
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
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

def train_loop(encoder, decoder, opt, loss_fn, data_loader):
    encoder.train()
    decoder.train()
    total_loss = 0
    counter = 0
    len_dataloader = len(data_loader)
    
    for batch in data_loader:
        states, actions, _, rewards, _, _ = batch

        X = torch.cat((states[:,:-1], actions[:,:-1], rewards[:,:-1]), dim=2).to(device)
        Y = torch.cat((states[:,-1], rewards[:,-1]), dim=1).to(device)

        encoding = encoder(X)
        pred = decoder(encoding)
        pred = pred.squeeze(0)

        # Permute pred to have batch size first again
        loss = loss_fn(pred[:,:-1], Y[:,:-1]) + loss_fn(pred[:,-1], Y[:,-1])

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
        if counter % 2000 == 0:
            print(f"[{counter}/{len_dataloader}]")
        counter += 1
        
    return total_loss / len(data_loader)

def validation_loop(encoder, decoder, loss_fn, data_loader):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            states, actions, _, rewards, _, _ = batch

            X = torch.cat((states[:,:-1], actions[:,:-1], rewards[:,:-1]), dim=2).to(device)
            Y = torch.cat((states[:,-1], rewards[:,-1]), dim=1).to(device)

            encoding = encoder(X)
            pred = decoder(encoding)
            pred = pred.squeeze(0)

            # Permute pred to have batch size first again
            loss = loss_fn(pred[:,:-1], Y[:,:-1]) + loss_fn(pred[:,-1], Y[:,-1])
            total_loss += loss.detach().item()
        
    return total_loss / len(data_loader)

def predict(encoder, decoder, data_loader, sample_count=5):
    encoder.eval()
    decoder.eval()
    total_loss = 0

    with torch.no_grad():
        for _ in range(sample_count):
            sample = next(iter(data_loader))
            tensors_to_concatenate = [sample[key] for key in include_input]
            sample = torch.cat(tensors_to_concatenate, dim=-1)
            
            X, Y =  sample[:,:-1].to(device), sample[:,-1].to(device)

            encoding = encoder(X)
            pred = decoder(encoding)
            pred = pred.squeeze(0)

            idx = 0
            loss = 0.
            for key in include_output:
                key_loss = loss_fn(pred[:,idx:idx+element_sizes[key]], Y[:,idx:idx+element_sizes[key]])
                loss += key_loss
                idx += element_sizes[key]
            total_loss += loss.detach().item()

            print(f"Predicition: \n{pred} ")
            print(f"Target: \n{X}")

    return total_loss / sample_count

def embedding_valdation(encoder, samples_num, data_loader, save_path):
    encoder.eval()
    
    counter = 0
    z_arr, task_arr, task_arm_arr, reward_arr = [], [], [], []
    with torch.no_grad():
        for batch in data_loader:
            states, actions, _, rewards, task_obs, task_arm = batch
            counter += states.shape[0]

            X = torch.cat((states[:,:-1], actions[:,:-1], rewards[:,:-1]), dim=2).to(device)

            task_arr.append(task_obs)
            task_arm_arr.append(task_arm)

            reward_arr.append(torch.sum(rewards, dim=1))
           
            encoding = encoder(X)
            encoding = encoding.squeeze(0)

            z_arr.append(encoding)

            if counter>samples_num:
                break
    task = torch.cat(task_arr).cpu().detach().numpy()
    task_arm = torch.cat(task_arm_arr).cpu().detach().numpy()
    rewards = torch.cat(reward_arr).cpu().detach().numpy()
    z = torch.cat(z_arr).cpu().detach().numpy()
    print("Saving embeddings...")
    torch.save({"task": task, "state_emb":z, "rewards": rewards, "task_arm": task_arm}, save_path)

########################### Hyperparameter ###########################

state_dim = 21
action_dim = 4
input_dim = state_dim + action_dim + 1
output_dim = state_dim + 1

hidden_size = 128 #64

batch_size = 128
sequence_length = 16
epochs = 0
sequence_len = 20

embedding_num = 16_000
predict_samples_num = 0
should_continue = True
should_safe = True
evaluate_embedding = True
use_gru = False
device = "cuda" if torch.cuda.is_available() else "cpu"
seed=1
model_path_rnn = 'Transformer_RNN/checkpoints/rnn_checkpoint.pth'
model_path_gru = 'Transformer_RNN/checkpoints/gru_checkpoint.pth'
dataset_path = 'Transformer_RNN/decision_tf_dataset/recorded_envs/'
val_dataset_path = 'Transformer_RNN/decision_tf_dataset/recorded_envs_val/'
embeddings_path = 'Transformer_RNN/embedding_log/rnn_emb.pth'
log_path = 'Transformer_RNN/tensorboard_log'

######################################################################


if __name__ == "__main__":
    set_seed(seed)
    loaded_dataset = TFDataset.load(dataset_path, sequence_length=sequence_len) # here one more
    loaded_val_dataset = TFDataset.load(val_dataset_path, sequence_length=sequence_len)
    total_size = len(loaded_val_dataset)
    val_size = int(0.9 * total_size)
    test_size =  total_size - val_size

    # create Dataloader
    val_dataset, test_dataset = random_split(loaded_val_dataset, [val_size, test_size])
    train_loader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if use_gru:
        encoder = GRUEncoder(input_dim, hidden_size).to(device)
        decoder = Predictor(hidden_size, output_dim).to(device)
        if should_continue:
            load_model(encoder, decoder, model_path_gru)
    else:
        encoder = LSTMEncoder(input_dim, hidden_size).to(device)
        decoder = Predictor(hidden_size, output_dim).to(device)
        if should_continue:
            load_model(encoder, decoder, model_path_rnn)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    loss_fn = nn.MSELoss(reduction='mean')

    writer = SummaryWriter(log_dir=log_path)

    print("Training and validating model")
    start_time = time.time()
    train_loss_list, validation_loss_list = [], []
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(encoder, decoder, opt, loss_fn, train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(encoder, decoder, loss_fn, val_loader)
        writer.add_scalar('Loss/validation', validation_loss, epoch)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f} | Time: {time.time() - start_time}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

        if should_safe:
            if use_gru:
                save_model(encoder, decoder, model_path_gru)
            else:
                save_model(encoder, decoder, model_path_rnn)

        start_time = time.time()

    if predict_samples_num > 0:
        predict(encoder, decoder, test_loader, predict_samples_num)

    if evaluate_embedding:
        embedding_valdation(encoder, embedding_num, val_loader, embeddings_path)

    writer.close()
