import numpy as np
import torch
import torch.nn.functional as F
from models import AllModels, setup_configs
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

eps = 0.5

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

configs_to_run = setup_configs()

def latent_perturb_tracts(configs_to_run, tract1, eps=1e-3):

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


        _, latent1, _, _ = m._vq_vae(m._pre_vq_conv(m._encoder(tract1)))


        interp = []
        for alpha in np.linspace(-1,1,9):
            between_z = latent1 + alpha * eps
            recon_between = m._decoder(between_z)
            interp.append(recon_between.squeeze(0).permute(1,0,2).detach().cpu().numpy())
        interp = np.array(interp)
        np.save(f'./latent_perturb/{m.model_used()}_MCP', interp)

        # Free up memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

latent_perturb_tracts(configs_to_run, tracts['sub-1175'][0][1].to(device).unsqueeze(0), eps=eps)

# Save Diagrams
ae = np.load('./latent_perturb/ae_MCP.npy')
vae = np.load('./latent_perturb/vae_MCP.npy')
vq = np.load('./latent_perturb/vq_MCP.npy')
vq_ema = np.load('./latent_perturb/vq_ema_MCP.npy')
vq_diff = np.load('./latent_perturb/vq_diff_MCP.npy')

# Architectures
archs = [(ae, 'ae'), (vae, 'vae'), (vq, 'vq'), (vq_ema, 'vq_ema'), (vq_diff, 'vq_diff')]

# Title settings with bold, dark and 14pt font
title_font = {
    'fontsize': 16,
    'fontweight': 'bold',
    'color': 'black'
}

n_rows = len(archs)
n_cols = 9
fig = plt.figure(figsize=(20, 10))
for row_num, (interp, label) in enumerate(archs):
    for col_num in range(n_cols):
        ax = fig.add_subplot(n_rows, n_cols, row_num * n_cols + col_num + 1, projection='3d')
        for j in range(64):
            if col_num == 4:
                x, y, z = interp[col_num, j]
                ax.plot(x, y, z, color='#4C72B0', alpha=0.25)
            else:
                x, y, z = interp[col_num, j]
                ax.plot(x, y, z, color='#C44E52', alpha=0.25)
        
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.zaxis._axinfo["grid"]['linewidth'] = 0.0
        ax.set_axis_off()
        
        if col_num == 4:
            ax.set_title(f'{label.upper()} - Recon', fontdict=title_font)
        else:
            ax.set_title(f'{col_num-4}ε', fontdict=title_font)

plt.tight_layout()
plt.savefig("./diagrams/perturb_MCP.pdf",format='pdf', dpi=800) 
plt.show()