"""Ntigkaris Alexandros"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from constants import *

class AQY_DS(Dataset):

      def __init__(
                    self,
                    df,
                    ):

            self.features = df[FEATURES].values
            self.target = df[TARGET].values

      def __len__(self):
            return len(self.features)

      def __getitem__(self,i):
            return (torch.tensor(self.features[i]).reshape(1,-1),
                    torch.tensor(self.target[i]).reshape(1,-1))

class AQY_NN(nn.Module):

    def __init__(
                  self,
                  input_size:int,
                  output_size:int,
                  hidden_size:int,
                ):

        super(AQY_NN,self).__init__()
        self.net = nn.Sequential(
                               nn.Linear(input_size,hidden_size,bias=True,),
                               nn.ReLU(),
                               nn.Linear(hidden_size,output_size,bias=True,),
                               )
    def forward(self,x):
        return self.net(x)

