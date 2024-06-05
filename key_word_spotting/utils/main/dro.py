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

class DRO():
    def __init__(self, N=20, writer=None, sigma=0.5):
        self.sigma = sigma
        self.N = N
        self.lr = 0.1
        self.writer = writer
        self.gamma = 2

    def requires_grad_(self, model:torch.nn.Module, index:int) -> None:
        for i, param in enumerate(model.parameters()):
            if i == index:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def forward(self, inputs, labels, model, trans, idx):
        # torch.autograd.set_detect_anomaly(True)
        inputs_prime = inputs.data.clone().to(device)
        optimizer_alpha = torch.optim.SGD(trans.parameters(), lr=self.lr)
        class_criterion = nn.CrossEntropyLoss()
        semantic_distance_criterion = nn.MSELoss()
        res = []
        for index in range(trans.count()):
            init_feature = None
            for n in range(self.N):
                optimizer_alpha.zero_grad()

                after_spec = trans(inputs_prime, index, use_noise=False)

                last_features = model(after_spec, get_feat=True)
                if n == 0:
                    init_feature = last_features.clone()
                rho = semantic_distance_criterion(last_features, init_feature)  #E[c(Z,Z0)]
                class_output = model(after_spec)
                
                loss_zt= class_criterion(class_output,labels.to(device))
                loss_phi = - (loss_zt - self.gamma * rho)                        # phi(theta,z0)
                loss_phi.backward(retain_graph=True)

                # if (trans.alphas[index] == 0).any():
                #     trans.alphas[index].grad[trans.alphas[index].grad == 0] = torch.randn_like(trans.alphas[index].grad[trans.alphas[index].grad == 0])
                # if idx == 139:
                #     print(index)
                #     print(f"{loss_phi.item()} - {torch.min(trans.alphas[index])} -> {torch.max(trans.alphas[index])}")

                optimizer_alpha.step()

                # trans.alphas[index].data.clamp_(-5, 1) 
                trans.alphas[index].data.clamp_(-5, 1) 
            # self.writer.add_scalar("Loss/max", loss_phi.item(), idx)
            res.append(after_spec)
        return res
