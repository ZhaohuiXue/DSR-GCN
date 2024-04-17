import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,A:torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1= torch.nn.Parameter(torch.tensor([0.5],requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 =nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 =nn.Sequential( nn.Linear(input_dim, output_dim))
        nodes_count=self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask=torch.ceil( self.A*0.00001)
        
        
    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat
    
    def forward(self, H, model='normal'):
        # # 方案一：minmax归一化
        H = self.BN(H)
        H_xx1= self.GCN_liner_theta_1(H)
        A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.03) * self.A + self.I
        # A = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())) * self.A + self.I
        # A=torch.clamp(A,0.1)
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)

        return output, A


class GCNLayersmall(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayersmall, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask = torch.ceil(self.A * 0.00001)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        # # 方案一：minmax归一化
        H = self.BN(H)
        H_xx1 = self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.03) * self.A + self.I
        A = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())) * self.A + self.I
        # A=torch.clamp(A,0.1)
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)
        return output, A


class DSR_GCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,Qsmall:torch.Tensor,Asmall:torch.Tensor):
        super(DSR_GCN, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.Qsmall = Qsmall
        self.Asmall = Asmall
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.norm_col_Qsmall = Qsmall / (torch.sum(Qsmall, 0, keepdim=True))
        layers_count=2
        self.GCN_Branch=nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branch.add_module('GCN_Branch'+str(i),GCNLayer(self.channel, 128, self.A))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 64, self.A))
        self.sigma2 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.GCN_Branchsmall=nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branchsmall.add_module('GCN_Branch'+str(i),GCNLayersmall(self.channel, 128, self.Asmall))
            else:
                self.GCN_Branchsmall.add_module('GCN_Branch' + str(i), GCNLayersmall(128, 64, self.Asmall))
        self.Softmax_linear =nn.Sequential(nn.Linear(128, self.class_count))
        self.Softmaxlinear = nn.Sequential(nn.Linear(64, self.class_count))

    def forward(self, x: torch.Tensor,y=torch.tensor(np.zeros(9)),merge=False):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape
        clean_x_flatten=x.reshape([h * w, -1])

        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 低频部分
        H1 = superpixels_flatten
        for i in range(len(self.GCN_Branch)): H1, _ = self.GCN_Branch[i](H1)
        GCN_resultbig = torch.matmul(self.Q, H1)  # 这里self.norm_row_Q == self.Q
        #
        superpixels_flattensmall = torch.mm(self.norm_col_Qsmall.t(), clean_x_flatten)  # 低频部分
        H2 = superpixels_flattensmall
        for i in range(len(self.GCN_Branchsmall)): H2, _ =self.GCN_Branchsmall[i](H2)
        GCN_resultsmall = torch.matmul(self.Qsmall, H2)

        if merge == True:  # 两组特征融合(两种融合方式)
            Y1 = torch.cat([GCN_resultbig, y], dim=-1)
            Y2 = torch.cat([GCN_resultsmall, y], dim=-1)
            Y = self.Softmax_linear(self.sigma2 * Y1 + (1 - self.sigma2) * Y2)
            Y = F.softmax(Y, -1)
            big = self.Softmaxlinear(GCN_resultbig)
            small = self.Softmaxlinear(GCN_resultsmall)
            loss = torch.square(big - small)
        return Y, loss


