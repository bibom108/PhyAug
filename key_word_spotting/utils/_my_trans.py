from enum import Enum
import hashlib
import math
import os
import random
import re
import time

from chainmap import ChainMap
from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pcen

device = torch.device("cuda")

class Transformation(nn.Module):
    def __init__(self, tf=None, use_trans=True, sigma=0.5):
        super(Transformation, self).__init__()
        self.n_mels = 40
        self.dct_filters = librosa.filters.dct(40, self.n_mels)
        self.sr = 16000
        self.f_max = 4000
        self.f_min = 20
        self.n_fft = 480
        self.hop_length = 16000 // 1000 * 10
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, trainable=True)
        self.sigma = sigma
        self.use_trans = use_trans
        self.N = 10
        self.lr = 0.01
        self.tf = [[torch.unsqueeze(torch.from_numpy(tf[i][0]).to(device), -1), 
                    torch.unsqueeze(torch.from_numpy(tf[i][1]).to(device), -1), 
                    torch.from_numpy(tf[i][2]).to(device)] 
                    for i in range(len(tf))
                    ]

    def forward(self, data):
        res = None
        for i in range(data.shape[0]):
            data_wo_batch = data[i]
            # STFT
            data_wo_batch = data_wo_batch.detach().cpu().numpy()
            data_wo_batch = np.abs(librosa.stft(data_wo_batch,
                                hop_length=self.hop_length,
                                n_fft=self.n_fft))

            # MAIN
            data_wo_batch = torch.from_numpy(data_wo_batch).to(device)
            loss_function = nn.CrossEntropyLoss()
            random.seed(time.clock())
            tf_index = random.randint(0,4)
            alpha = torch.randn_like(self.tf[tf_index][0]).to(device)
            alpha = Variable(alpha, requires_grad=True)
            optimizer_alpha = torch.optim.SGD([alpha], lr=0.01)
            for i in range(self.N):
                # delta = (torch.randn_like(alpha) * self.sigma).to(device)
                optimizer_alpha.zero_grad()
                a = torch.exp(alpha)*self.tf[tf_index][0]
                b = self.tf[tf_index][1]
                loss = loss_function(a, b)
                loss.backward()
                # avoid nan or inf if gradient is 0
                if (alpha.grad == 0).any():
                    alpha.grad[alpha.grad == 0] = torch.randn_like(alpha.grad[alpha.grad == 0])
                optimizer_alpha.step()
            delta = (torch.randn_like(alpha) * self.sigma).to(device)
            data_wo_batch = torch.exp(alpha + delta) * self.tf[tf_index][2] * data_wo_batch
            data_wo_batch = data_wo_batch.detach().cpu().numpy()

            # melspectrogram
            data_wo_batch = data_wo_batch**2
            data_wo_batch = librosa.feature.melspectrogram(S=data_wo_batch,
                                                sr=self.sr,
                                                fmin=self.f_min,
                                                fmax=self.f_max,
                                                n_mels=self.n_mels)
            data_wo_batch[data_wo_batch > 0] = np.log(data_wo_batch[data_wo_batch > 0])
            data_wo_batch = [np.matmul(self.dct_filters, x) for x in np.split(data_wo_batch, data_wo_batch.shape[1], axis=1)]
            data_wo_batch = np.array(data_wo_batch, order="F").astype(np.float32)
            mean = np.mean(data_wo_batch)
            std = np.std(data_wo_batch)
            data_wo_batch = (data_wo_batch - mean)/std

            audio_tensor = torch.from_numpy(data_wo_batch.reshape(1, -1, 40))

            res = audio_tensor if res is None else torch.cat((res, audio_tensor), 0)

        return res