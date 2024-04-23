import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def dismap(x, name='pred'):
    
    x = x.data.cpu().numpy()
    x = x.mean(1)
    for j in range(x.shape[0]):
        plt.cla()
        y = x[j]
        df = pd.DataFrame(y)
        sns.heatmap(df)
        plt.savefig('results/dismap/{}_{}.png'.format(name,str(j)))
        plt.close()
    return True