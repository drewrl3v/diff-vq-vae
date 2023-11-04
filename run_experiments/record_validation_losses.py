import numpy as np
import torch
import torch.nn.functional as F
from models import AllModels
from dipy.segment.bundles import bundle_shape_similarity
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
rng = np.random.RandomState()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_set = torch.load('data_ml/val_set.pt')

# Collect corresponding tracts
tracts = {}

for i, (tract, label) in enumerate(val_set):
    _, name_of_tract = label.split('__')

    if name_of_tract in tracts:
        tracts[name_of_tract].append(tract)
    else:
        tracts[name_of_tract] = []
        tracts[name_of_tract].append(tract)

# Test all 6 models
print('Configuring Models')
configs_to_run = [[False for _ in range(5)] for _ in range(5)]
for i in range(5): 
    for j in range(5):
        if j == i:
            configs_to_run[i][j] = True

def losses_all_tracts(configs_to_run, tracts):
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

        for name_of_tract in tracts.keys():
            recon_mse_losses = []
            recon_buan_losses = []
            data = tracts[name_of_tract]
            print(m.model_used(), name_of_tract)
            for x in data:
                x = x.to(device).unsqueeze(0)
                vq_loss, x_recon, perplexity, encodings = m(x)
                recon_mse_losses.append(F.mse_loss(x, x_recon).detach().cpu().numpy())

                # formate for buan
                x = x.squeeze(0).permute(1,2,0).detach().cpu().numpy()
                x = [x[i] for i in range(64)]
                x_recon = x_recon.squeeze(0).permute(1,2,0).detach().cpu().numpy()
                x_recon = [x_recon[i] for i in range(64)]
                recon_buan_losses.append(bundle_shape_similarity(x, x_recon, rng, [0], threshold = 0.05))
            recon_mse_losses = np.array(recon_mse_losses)
            recon_buan_losses = np.array(recon_buan_losses)
            np.save(f'./mse_losses/{name_of_tract}_{m.model_used()}.npy', recon_mse_losses)
            np.save(f'./buan_losses/{name_of_tract}_{m.model_used()}.npy', recon_buan_losses)

        # Free up memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

losses_all_tracts(configs_to_run, tracts)