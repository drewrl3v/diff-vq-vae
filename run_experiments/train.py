''' 
We Train all the models here.
Statistics such as latent embeddings, interpolations, ect.
are computed in a separate file after the model is trained.
'''
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from scipy.signal import savgol_filter
from models import AllModels, setup_configs
from streamline_utils import NewStreamlineDataset
import numpy as np
import gc

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load data, scale and format
print('Loading Data')
all_data = torch.load('./data_ml/processed_tracts.pt')

# This class will standardize the streamlines to all be within a [-1,1]^3 unit cube
streamlines = NewStreamlineDataset(all_data)

# Static Training Parameters across all networks
batch_size = 256
num_training_updates = 30_000
num_epochs = 1

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99
tau = 10.0
std = 2.0

learning_rate = 1e-3

# 90% Train, 10% Validation
torch.manual_seed(1337)
start = int(0.9 * len(streamlines))
stop = len(streamlines) - start
train_set, val_set = torch.utils.data.random_split(streamlines, [start, stop])
# We save this specific validation set for computing statistics later
torch.save(val_set, './data_ml/val_set.pt')

# Train and validation loaders
torch.manual_seed(1337)
training_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle=True, drop_last=True)
validation_loader = data.DataLoader(val_set, batch_size=1, shuffle=True, drop_last=False)

# Test all 5 models
print('Configuring Models')
configs_to_run = setup_configs()

def train(configs_to_run):
    for config in configs_to_run:
        # current configuration
        ae = config[0]; vae = config[1]; vq=config[2]; vq_ema = config[3]; vq_diff = config[4]

        # Setup model
        model = AllModels(num_hiddens, num_residual_layers, num_residual_hiddens,
                    num_embeddings, embedding_dim, commitment_cost, tau, std, decay, 
                    ae, vae, vq, vq_ema, vq_diff
                    ).to(device)
        model.train()
        type_of_model = model.model_used()
        print(f"Currently Training: {type_of_model}")

        # Setup Optimizer:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

        # Keep a record of the MSE loss (and the perplexity if the model has perplexity)
        train_res_recon_error1 = []
        train_res_perplexity1 = []

        for epoch in range(num_epochs):
            for i in range(num_training_updates):

                (data, _) = next(iter(training_loader))
                B, fibers, dim, points = data.shape
                data = data.float().to(device)
                    
                optimizer.zero_grad()
                vq_loss, data_recon, perplexity, encodings = model(data)
                recon_error = F.mse_loss(data_recon, data)
                loss = recon_error + vq_loss
                loss.backward()
                optimizer.step()
                
                if epoch == 0:
                    train_res_recon_error1.append(recon_error.item())
                    if perplexity:
                        train_res_perplexity1.append(perplexity.item())

                    if (i+1) % 100 == 0:
                        print(f'{i + 1} iterations')
                        print(f'recon_error: {np.mean(train_res_recon_error1[-100:]):.5f}')
                        if perplexity:
                            print(f'perplexity: {np.mean(train_res_perplexity1[-100:]):.5f}')
                        print()

        # Save the recon erros (and perplexity if the model is a vq model)
        train_res_recon_error1 = np.array(train_res_recon_error1)
        np.save(f'./data_ml/{type_of_model}_loss.npy', train_res_recon_error1)
        if type_of_model in ['vq', 'vq_ema', 'vq_dff']:
            train_res_perplexity1 = np.array(train_res_perplexity1)
            np.save(f'./data_ml/{type_of_model}_perplexity.npy', train_res_perplexity1)
        
        # Save the trained model
        torch.save(model.state_dict(), f'./saved_models/{type_of_model}.pt')

        # Free up memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

print('Training Models')
train(configs_to_run)