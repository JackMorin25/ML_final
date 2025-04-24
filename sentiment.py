import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
df = pd.read_csv('data/IMDB Dataset.csv')



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")





