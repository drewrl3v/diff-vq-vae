import os
import numpy as np
import torch 
import pandas as pd
import matplotlib.pyplot as plt

def list_files(directory):
    try:
        # Get the list of all files and directories in the specified directory
        files_and_dirs = os.listdir(directory)
        
        # Filter out directories, only keep files
        files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
        
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

buan_losses = list_files('./buan_losses/')

# Construct a Datafram with the BUAN scores
data = {}
for file in buan_losses:
    score = np.load(f'./buan_losses/{file}')
    mean, std = np.mean(score), np.std(score)
    label, architecture = file.split('.')[0].split('_')[:-1], file.split('.')[0].split('_')[-1]
    if 'vq' in label:
        label.remove('vq')
    label = '_'.join(label)

    if architecture in data:
        data[architecture].append([label, f'{mean:.6f} ± {std:.6f}'])
    else:
        data[architecture] = [[label, f'{mean:.6f} ± {std:.6f}']]

for arch in data.keys():
    data[arch].sort(key=lambda x:x[0], reverse=True)

table = {'Name': [x[0] for x in data['ae']],
        'AE': [x[1] for x in data['ae']]}

df = pd.DataFrame(table)
df.to_csv('./diagrams/buan_table.csv', index = False)