
import librosa
import numpy as np
import torch
import torch.nn as nn
import pcen
from MelScale import MelScale

device = torch.device('cuda')

class Transformation(nn.Module):
    def __init__(self, tf, sigma, alphas=None):
        super(Transformation, self).__init__()
        self.n_mels = 40
        self.dct_filters = torch.from_numpy(librosa.filters.dct(40, self.n_mels).astype(np.float32)).to(device)
        self.sr = 16000
        self.f_max = 4000
        self.f_min = 20
        self.n_fft = 480
        self.hop_length = 16000 // 1000 * 10
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, trainable=True)
        self.sigma = sigma
        self.melscale_transform = MelScale(sample_rate=16000, f_min=20, f_max=4000, n_mels =40, n_stft =241, norm ="slaney")
        # self.tf = [[torch.unsqueeze(torch.from_numpy(tf[i][0]).to(device), -1), 
        #             torch.unsqueeze(torch.from_numpy(tf[i][1]).to(device), -1), 
        #             torch.from_numpy(tf[i][2]).to(device)] 
        #             for i in range(len(tf))
        #             ]
        self.tf = [torch.from_numpy(tf[i]).to(device)for i in range(len(tf))]
        self.alphas = nn.ParameterList([torch.nn.Parameter(torch.randn_like(self.tf[i][2])) for i in range(len(tf))]) 

    def count(self):
        return len(self.tf)
    
    def forward(self, inputs, index, use_noise=False):
        # inputs is tensor with batch
        noise = torch.rand_like(self.tf[index]) * self.sigma if use_noise else 0
        data = torch.exp(self.alphas[index] + noise) * self.tf[index] * inputs
        data = data**2
        # compute mfcc
        data = self.melscale_transform(data)
        data[data > 0] = torch.log(data[data > 0])
        data = [torch.matmul(self.dct_filters, x) for x in torch.split(data, data.shape[1], dim=1)][0]
        ## adding z-score normalize
        mean = torch.mean(data)
        std = torch.std(data)
        if std != 0:
            data = (data - mean)/std
        data = torch.transpose(data, 1, 2)
        return data
