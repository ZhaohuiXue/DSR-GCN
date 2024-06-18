import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def transpose(x, source='NHWC', target='NCHW'):#将数据集设置为N channel h width
    return x.transpose([source.index(d) for d in target])
class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):

        c = self.BN(input)
        out = self.point_conv(c)
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

# class CNN(nn.Module):
#     def __init__(self, height: int, width: int, changel: int, class_count: int):
#         super(CNN, self).__init__()
#         # 类别数,即网络最终输出通道数
#         self.class_count = class_count  # 类别数
#         # 网络输入数据大小
#         self.channel = changel
#         self.height = height
#         self.width = width
#         layers_count = 2
#         # Pixel-level Convolutional Sub-Network
#         self.CNN_Branch = nn.Sequential()
#         for i in range(layers_count):
#             if i < layers_count - 1:
#                 self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(30, 128, kernel_size=5))
#             else:
#                  self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))
#         # Softmax layer
#         # self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))
#         global size
#         global t
#         size, t = self._get_final_flattened_size()
#
#         self.Softmax_linear = nn.Sequential(nn.Linear(size, self.class_count))
#     def _get_final_flattened_size(self):
#         with torch.no_grad():
#             x = torch.zeros(
#                 (1,  self.channel, self.height, self.width)
#             )
#         x = self.CNN_Branch(x)
#         t, c, w, h = x.size()
#
#         return c * w * h, t
#     def forward(self, x: torch.Tensor,y=torch.tensor(np.zeros(9)),merge =False):
#         '''
#         :param x: H*W*C
#         :return: probability_map
#         '''
#         # x = transpose(x)
#         (n, h, w, c) = x.shape
#         # CNN_result = self.CNN_Branch(torch.unsqueeze(x.permute([0, 3, 1, 2]), 1))  # spectral-spatial convolution
#         CNN_result = self.CNN_Branch(x.permute([0, 3, 1, 2]))
#         # CNN_result = CNN_result.permute([0,2,1,3]).reshape([16, h*w*64])
#         CNN_result = CNN_result.reshape([16, size])
#         # 两组特征融合(两种融合方式)
#         if merge == True:
#             Y = torch.cat([y, CNN_result], dim=-1)
#             Y = self.Softmax_linear(Y)
#             Y = F.softmax(Y, -1)
#         elif merge == False:
#             Y = self.Softmax_linear(CNN_result)
#             Y = F.softmax(Y, -1)
#         return Y
class CNN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int):
        super(CNN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        layers_count = 2

        self.CNN_Branch1= SSConv(changel, 128, kernel_size=7)
        self.CNN_Branch2 = SSConv(128, 64, kernel_size=7)
        # self.CNN_Branch3 = SSConv(64, 16, kernel_size=5)
                # self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5,))
        # self.CNN_Branch5=SSConv(32, 16, kernel_size=5)
        # Softmax layer
        # self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))
        # global size
        # global t
        # size, t = self._get_final_flattened_size()

        # self.Softmax_linear = nn.Sequential(nn.Linear(1296, self.class_count))
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.Softmax_linear =nn.Linear(128, self.class_count)
        self.projector = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
        )



        # return t* w * h, t
    def forward(self, x: torch.Tensor,full=False,contrast=False):
        '''
        :param x: H*W*C
        :return: probability_map
        '''

        if full == False:
            CNN_result = self.CNN_Branch1(x)
            CNN_result = self.CNN_Branch2(CNN_result)
            CNN_result = self.pool(CNN_result)
            n, h, w, c = CNN_result.shape
            CNN_result = CNN_result.reshape((-1, h * w * c))
            Y = self.projector(CNN_result)
            CNN_result = self.Softmax_linear(Y)
        elif full == True:
            x = torch.unsqueeze(x.permute([2, 0, 1]), 0)
            CNN_result = self.CNN_Branch1(x)
            CNN_result = self.CNN_Branch2(CNN_result)
            n, c, w, h = CNN_result.shape
            CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])
            Y = CNN_result

        return Y,CNN_result