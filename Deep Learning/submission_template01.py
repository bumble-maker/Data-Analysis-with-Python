import numpy as np
import torch
from torch import nn

def create_model():
    # Linear layer mapping from 784 features, so it should be 784->256->16->10
    layer1 = nn.Linear(784,256, bias = True)
    layer2 = nn.Linear(256,16, bias = True)
    layer3 = nn.Linear(16,10, bias = True)
    # your code here
    NN = nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), layer3)
    # return model instance (None is just a placeholder)
    return NN

def count_parameters(model):
    # your code here
    total = sum(p.numel() for p in model.parameters())
    # верните количество параметров модели model
    return total
