import numpy as np
import torch
from models import AllModels, setup_configs
import gc

# Parameters for every model
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

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Validatoin Data
val_set = torch.load('data_ml/val_set.pt')

# Collect corresponding tracts
tracts = {}
for i, (tract, label) in enumerate(val_set):
    subject, name_of_tract = label.split('__')
    if subject in tracts:
        tracts[subject].append([name_of_tract, tract])
    else:
        tracts[subject] = []
        tracts[subject].append([name_of_tract, tract])

# Since the data-split between test and validation is not perfect there 
# is inevitably some subjects in this dataset that do not have all
# streamline groups, so we order by the complete ones first
num_tracts_per_sub = []
for subject in tracts.keys():
    hold = set()
    for name_of_tract, _ in tracts[subject]:
        hold.add(name_of_tract)
    num_tracts_per_sub.append((len(hold), subject))

num_tracts_per_sub.sort(key=lambda x:x[0], reverse=True)

# Model initialization
configs_to_run = setup_configs()

def reconstruct_full_tract(configs_to_run, subject):
    for config in configs_to_run:
        # current configuration
        ae = config[0]; vae = config[1]; vq=config[2]; vq_ema = config[3]; vq_diff = config[4]

        model = AllModels(num_hiddens, num_residual_layers, num_residual_hiddens,
            num_embeddings, embedding_dim, commitment_cost, tau, std, decay, 
            ae, vae, vq, vq_ema, vq_diff
            ).to(device)

        if ae:
            model.load_state_dict(torch.load('saved_models/ae.pt'))
        elif vae:
            model.load_state_dict(torch.load('saved_models/vae.pt'))
        elif vq:
            model.load_state_dict(torch.load('saved_models/vq.pt'))
        elif vq_ema:
            model.load_state_dict(torch.load('saved_models/vq_ema.pt'))
        else:
            model.load_state_dict(torch.load('saved_models/vq_diff.pt'))

        m = model.eval()
        print(m.model_used())

        recons = {}
        originals = {}
        for label, x in subject:

            # Keep a record of the original tract and express as numpy
            if label in originals:
                originals[label].append(x.permute(1,0,2).detach().cpu().numpy())
            else:
                originals[label] = [x.permute(1,0,2).detach().cpu().numpy()]
            
            # Then reconstruct
            x = x.to(device).unsqueeze(0)
            vq_loss, x_recon, perplexity, encodings = m(x)
            x_recon = x_recon.squeeze(0)

            # Keep a record of the recon tract and exress as numpy
            if label in recons:
                recons[label].append(x_recon.permute(1,0,2).detach().cpu().numpy())
            else:
                recons[label] = [x_recon.permute(1,0,2).detach().cpu().numpy()]

        # concatenate the full reconstruction 
        for label in originals.keys():

            originals[label] = np.concatenate(originals[label])
            np.save(f'./subject_originals/{label}.npy', originals[label])

            recons[label] = np.concatenate(recons[label])
            np.save(f'./subject_recons/{label}_{m.model_used()}.npy', recons[label])


        # Free up memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

reconstruct_full_tract(configs_to_run, num_tracts_per_sub[0])