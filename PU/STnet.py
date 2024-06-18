import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class STnet(nn.Module):
    def __init__(self, changel: int,outchannel:int):
        super(STnet, self).__init__()
        self.channel = changel
        self.outchannel = outchannel
        layers_count = 2
        # Spectra Transformation Sub-Network
        self.denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, self.outchannel, kernel_size=(1, 1)))
                self.denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.outchannel), )
                self.denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(self.outchannel, self.outchannel, kernel_size=(1, 1)))
                self.denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
    def forward(self, x: torch.Tensor,full = False):
        if full == False:
            # x = x.permute([0, 3, 1, 2])
            x = self.denoise(x)

        elif full == True:
            x = torch.unsqueeze(x.permute([2, 0, 1]), 0)
            x = self.denoise(x)
            # x = torch.squeeze(x, 0).permute([1, 2, 0])
            x = torch.squeeze(x, 0).permute([1, 2, 0])
        return x