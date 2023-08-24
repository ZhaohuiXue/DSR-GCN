import torch

from sklearn.decomposition import PCA

import numpy as np
import scipy.io as sio
import os
import random
import copy
from torchvision import transforms

def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    height, width,_=gt.shape
    GT_One_Hot = []  # 转化为one-hot形式的标签
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count,dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    # indexX = np.arange(0, newX.shape[0], 1)
    pca = PCA(n_components=numComponents, whiten=True)#whiten是否白化，使得每个特征有相同的方差
    newX = pca.fit_transform(newX)#训练pca，让特征在前几维度
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    # indexX = np.reshape(indexX, (X.shape[0], X.shape[1]))
    return newX, pca

def mnf(x, n_components=75, whiten=False):
    # Maximun Noise Fraction
    # Refer: A Transformation for Ordering Multispectral Data in Terms of Image Quality
    # with Implications for Noise Removal. TGRS 1987.
    shp = x.shape
    data = np.transpose(np.reshape(x, [shp[0] * shp[1], shp[2]]), [1, 0])
    data_norm = data - np.mean(data, 1, keepdims=True)
    sigma_x = np.cov(data_norm)

    n1 = np.zeros(shape=[shp[0], shp[1], shp[2]])
    t = x.astype(dtype=np.float32)
    for i in range(1, shp[0]):
        n1[i, :, :] = t[i, :, :] - t[i - 1, :, :]
    n2 = np.zeros(shape=[shp[0], shp[1], shp[2]])
    for i in range(0, shp[1] - 1):
        n2[:, i, :] = t[:, i, :] - t[:, i + 1, :]

    x_noise = (n1 + n2) / 2

    noise = np.transpose(np.reshape(x_noise, [shp[0] * shp[1], shp[2]]), [1, 0])
    noise_norm = noise - np.mean(noise, 1, keepdims=True)
    sigma_noise = np.cov(noise_norm)
    inv_sigma_noise = np.linalg.inv(sigma_noise)

    # generate eig_value and eig_vector
    [eig_value, eig_vectors] = np.linalg.eig(np.matmul(inv_sigma_noise, sigma_x))

    # project to a new column vector space
    data_mnf = np.dot(np.transpose(eig_vectors[0:n_components,:],(0,1)), data)
    # c1 = np.cov(data_mnf)
    # rescale each variable to unit variance.
    if whiten:
        cov1 = np.cov(data_mnf)
        s = np.diagonal(cov1)
        epison = 0
        scale = np.diag((1 / (np.sqrt(s + epison))))
        data_mnf = np.matmul(scale, data_mnf)
    # c2 = np.cov(data_mnf)
    x_mnf = np.transpose(data_mnf, [1, 0]).astype(dtype=np.float32)
    x_mnf = np.reshape(x_mnf, (shp[0], shp[1], n_components))
    return x_mnf
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
def neighbor_add(indexX, row, col, image_row, image_col, w_size=3):  # 给出 row，col和标签，返回w_size大小的cube
    neighbor_index = []
    t = w_size // 2
    for i in range(-t, t + 1):
        for j in range(-t, t + 1):
            if i + row < 0 or i + row >= image_row or j + col < 0 or j + col >= image_col:
                neighbor_index.append(indexX)
            else:
                neighbor_index.append(indexX+i*image_row+j)
    return neighbor_index
def createImageCubes(X,y,windowSize=5):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData=[]
    patchesLabels = []
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[0:2*margin + 1, c - margin:c + margin + 1]
            patchesData.append(patch)
            patchesLabels.append(y[r - margin, c - margin])

        # zeroPaddedX= np.delete(zeroPaddedX, 0, axis=0)
        zeroPaddedX = zeroPaddedX[1:,:]
    del zeroPaddedX

    patchesLabels = [i - 1 for i in patchesLabels]
    return patchesData, patchesLabels
def PerClassSplit(X, y,  perclass, stratify):

    label_train = []
    label_y_train = []
    unlabel_train = []
    unlabel_ytrain = []
    X_test = []
    y_test = []

    randomArray = list()
    for label in range(0, 9):
        index = [i for i in range(len(y)) if y[i] == label]
        index = np.array(index)
        n_data = index.shape[0]
        randomArray.append(np.random.permutation(n_data))

    # print('max：' + str(np.max(np.array(y))) + ', min: ' + str(np.min(np.array(y))))
    label_idx = []
    unlabel_idx = []
    test_idx = []

    for label in range(0, 9):
        index = [i for i in range(len(y)) if y[i] == label]
        index = np.array(index)
        n_data = index.shape[0]
        randomX = randomArray[label]
        label_idx.extend(index[randomX[0:20]])#20 别忘记出纯3DCNN的图做对比
        unlabel_idx.extend(index[randomX[20:200]])
        test_idx.extend(index[randomX[20:n_data]])
    for i in label_idx:
        label_train.append(X[i])
        label_y_train.append(y[i])
    for i in unlabel_idx:
        unlabel_train.append(X[i])
        unlabel_ytrain.append(y[i])
    for i in test_idx:
        X_test.append(X[i])
        y_test.append(y[i])


    return label_train, X_test, label_y_train, y_test, unlabel_train, unlabel_ytrain
def indexgetdata(index,data):
    labeled_data = []
    unlabeled_data = []
    testdata = []
    for i in range(3):
        if i == 0:
            for j in index[i]:
                labeled_data.append(data[j])
        # elif i == 1:
        #     for j in index[i]:
        #         unlabeled_data.append(data[j])
        elif i == 1:
            for j in index[i]:
                testdata.append(data[j])
    labeled_data = np.array(labeled_data)
    # unlabeled_data = np.array(unlabeled_data)
    testdata = np.array(testdata)
    return labeled_data,  testdata
def indexgetlabel(index,label,class_count,m):
    trainlabel = []
    # falselabel = []
    testlabel = []
    truetest = []
    for i in range(2):
        if i == 0:
            for j in index[i]:
                trainlabel.append(label[j])
                truetest.append(m[j])
        # elif i == 1:
        #     for j in index[i]:
        #         falselabel.append(label[j])
        elif i == 1:
            for j in index[i]:
                testlabel.append(label[j])
                truetest.append(m[j])
    trainlabel = np.array(trainlabel)
    # falselabel = np.array(falselabel)

    testlabel = np.array(testlabel)

    return trainlabel,testlabel
    
def feature_normalize1(data):
    mu = np.mean(data, 0)
    xu = np.std(data, 0)
    return (data - mu)/xu
def feature_normalize2(data):

    return (data - np.min(data, 0))/(np.max(data, 0)-np.min(data, 0))
class PairDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,Datapath1,Datapath2,Labelpath,trans):
        # 1. Initialize file path or list of file names.
        self.data=np.load(Datapath)
        # self.data2=np.load('Xtrain.npy')
        self.DataList1=np.load(Datapath1)
        self.DataList2 = np.load(Datapath2)
        self.LabelList=np.load(Labelpath)
        self.trans=trans
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data

        index=index
        num1=self.DataList1[index]

        Data=self.trans(self.data[num1].astype('float64'))
        # Data=Data.view(-1,Data.shape[0],Data.shape[1],Data.shape[2])
        num2=self.DataList2[index]
        Data2=self.trans(self.data[num2].astype('float64'))
        # Data2=Data2.view(-1,Data2.shape[0],Data2.shape[1],Data2.shape[2])
        Label=self.LabelList[index]
        return Data, Data2, Label
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.DataList1)
class SamplePairDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,Datapath1,Datapath2,Labelpath,trans):
        # 1. Initialize file path or list of file names.
        self.data=np.load(Datapath)
        # self.data2=np.load('Xtrain.npy')
        self.DataList1=np.load(Datapath1)
        self.DataList2 = np.load(Datapath2)
        self.LabelList=np.load(Labelpath)
        self.trans=trans
        un_matchlist = [i for i in range(len(self.LabelList)) if self.LabelList[i] == 0]
        print(len(un_matchlist))
        # semi_matchlist = [i for i in range(len(self.LabelList)) if self.LabelList[i] == 0.5]
        matchlist = [i for i in range(len(self.LabelList)) if self.LabelList[i] == 1]
        self.un_matchlist = random.sample(un_matchlist, int(len(un_matchlist)/10)) # 2
        # self.semi_matchlist = random.sample(semi_matchlist, int(self.output_units*(self.perclass-1)))
        self.matchlist = random.sample(matchlist, int(len(matchlist)/2)) # 1
        self.un_matchlist.extend(self.matchlist)
        random.shuffle(self.un_matchlist)
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        list_index = self.un_matchlist[index]
        num1 = self.DataList1[list_index]
        Data_1 = self.trans(self.data[num1].astype('float64'))
        # Data_1 = Data_1.view(-1, Data_1.shape[0], Data_1.shape[1], Data_1.shape[2])  # 变成4维干什么
        num2 = self.DataList2[list_index]
        Data_4 = self.trans(self.data[num2].astype('float64'))
        # Data_4 = Data_4.view(-1, Data_4.shape[0], Data_4.shape[1], Data_4.shape[2])
        Label = self.LabelList[list_index]
        return Data_1, Data_4, Label
        # index=index
        # num1=self.DataList1[index]
        #
        # Data=self.trans(self.data[num1].astype('float64'))
        # # Data=Data.view(-1,Data.shape[0],Data.shape[1],Data.shape[2])
        # num2=self.DataList2[index]
        # Data2=self.trans(self.data[num2].astype('float64'))
        # # Data2=Data2.view(-1,Data2.shape[0],Data2.shape[1],Data2.shape[2])
        # Label=self.LabelList[index]
        # # index=index
        # Data2=Data2.view(-1,Data2.shape[0],Data2.shape[1],Data2.shape[2])

        # return Data, Data2, Label
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.un_matchlist)
class MYDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,Labelpath,transform):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.Labellist=(np.load(Labelpath)).astype(int)
        self.transform=transform
    def __getitem__(self, index):


        index=index

        Data=self.transform(self.Datalist[index].astype('float64'))
        # Data=Data.view(1,Data.shape[0],Data.shape[1],Data.shape[2])
        return Data ,self.Labellist[index]
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)
def get_data(data,label,index,windowSize,classcount):
    #这里注意输入的y应该是onehot后的结果
    X, y = data, label
    m = label.reshape(-1)
    # if PCA == True:
    #     X, pca = applyPCA(X, numComponents=K)
    # else:
    #     X = X
    X,y = createImageCubes(X.astype('float32'),y, windowSize=windowSize)
    # stratify = np.arange(0, output_units, 1)
    # X = feature_normalize1(np.asarray(X).astype('float32'))
    K = 103
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(K), np.ones(K))
    ])

    labeled_data,testdata = indexgetdata(index,X)
    trainlabel,testlabel = indexgetlabel(index,y,classcount,m)
    datalist1 = []
    datalist2 = []
    labellist = []

    np.save('ytrain.npy', trainlabel)
    np.save('xtrain.npy', labeled_data)
    for order in range(len(trainlabel)):
        for k in range(len(trainlabel)):
            if not order == k:
                y = int(trainlabel[order] == trainlabel[k])
                datalist1.append(order)
                datalist2.append(k)
                labellist.append(y)
        if len(labellist) > 10000:
            print(len(labellist))
            break
    np.save('TrainData1.npy', datalist1)
    np.save('TrainData2.npy', datalist2)
    np.save('TrainLabel.npy', labellist)
    del datalist1, datalist2, labellist
    Datapath1 = 'TrainData1.npy'
    Datapath2 = 'TrainData2.npy'
    PairLabelpath = 'TrainLabel.npy'
    Datapath = 'xtrain.npy'
    labelpath = 'ytrain.npy'
    contrast_data = PairDataset(Datapath,Datapath1, Datapath2, PairLabelpath,trans)
    classidata  =  MYDataset(Datapath,labelpath,trans)
    # contrast_data = torch.FloatTensor(contrast_data)
    return contrast_data,classidata