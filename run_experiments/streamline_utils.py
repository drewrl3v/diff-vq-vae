import torch
import torch.nn as nn
import torch.utils.data as data

class NewStreamlineDataset(data.Dataset):
    ''' 
    Meant to Load an individual tract specified by the pad to the data
    '''
    def __init__(self, data_ls):
        super().__init__()
        all_data = []
        all_label = []

        # rescaling step for streamlines
        for (data, label) in data_ls:
            all_data.append(data.permute(1,0,2))
            all_label.append(label)
        self.data = torch.stack(all_data)
        ma = torch.max(self.data)
        mi = torch.min(self.data)
        self.data = 2 * ((self.data - mi) / (ma - mi)) - 1

        self.label = all_label

    def __len__(self):
        # number of data points we have
        return len(self.data) #.shape[0]
    
    def __getitem__(self, idx):
        # return the idx-th data point in the dataset
        data_point = self.data[idx]
        data_label = self.label[idx]
        return (data_point, data_label)