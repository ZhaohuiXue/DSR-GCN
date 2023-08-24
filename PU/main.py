import numpy as np
import scipy.io as sio
import torch
import random
from dataload import get_data
from sklearn import preprocessing
import CNNmodule
import DSRGCN
import PCA_SLIC
import STnet
from sklearn import metrics
from matplotlib import  pyplot as plt
from contrast_train import contrastlearning
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
samples_type = ['ratio', 'same_num'][1]
OAmean = []
for (FLAG, curr_train_ratio, Scale) in [(2, 0.1,500)]:
    # for t in [90]:
    for sampless in [5]:
        print(sampless)
        for t in [128]:
            winsize = 9
            torch.cuda.empty_cache()
            OA_ALL = []
            AA_ALL = []
            KPP_ALL = []
            AVG_ALL = []
            Train_Time_ALL = []
            Test_Time_ALL = []

            Seed_List = [10,1,2,3,4,15,6,7,8,12]  # 随机种子点


            if FLAG == 2:

                data_mat = sio.loadmat('/tmp/pycharm_project_767/HyperImage_data/paviaU/PaviaU.mat')
                data0 = data_mat['x']
                gt_mat = sio.loadmat('/tmp/pycharm_project_767/HyperImage_data/paviaU/Pavia_University_gt.mat')
                gt = gt_mat['y']
            #     # 参数预设
                # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
                val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
                class_count = 9  # 样本类别数
                learning_rate = 5e-4  # 学习率
                max_epoch = 600  # 迭代次数
                dataset_name = "paviaU_"  # 数据集名称
                # superpixel_scale = 100
                pass

            orig_data = data0
            height, width, bands = data0.shape
            data0 = np.reshape(data0, [height * width, bands])
            ddd = data0
            minMax = preprocessing.StandardScaler()
            GCNdata = minMax.fit_transform(data0)
            GCNdata = np.reshape(GCNdata, [height, width, bands])
            # feature_data = np.sum(GCNdata, 2)
            train_samples_per_class = sampless
            alllabelgt = gt
            gt_reshape = np.reshape(gt, [-1])
            a = np.min(gt_reshape)
            oa = []
            gtdraw = gt_reshape

            def draw_result(labels,gtt, dataID=1):
                # ID=1:Pavia University
                # ID=2:Indian Pines
                # ID=3:KSC
                labels = np.array(labels).reshape([-1])
                num_class = labels.max() + 1

                if dataID == 1:
                    row = 610
                    col = 340
                    palette = np.array([[0, 0, 255],
                                        [76, 230, 0],
                                        [255, 190, 232],
                                        [255, 0, 0],
                                        [156, 156, 156],
                                        [255, 255, 115],
                                        [0, 255, 197],
                                        [132, 0, 168],
                                        [0, 0, 0]])
                    palette = palette * 1.0 / 255
                elif dataID == 2:
                    row = 145
                    col = 145
                    palette = np.array([[0, 168, 132],
                                        [76, 0, 115],
                                        [0, 0, 0],
                                        [190, 255, 232],
                                        [255, 0, 0],
                                        [115, 0, 0],
                                        [205, 205, 102],
                                        [137, 90, 68],
                                        [215, 158, 158],
                                        [255, 115, 223],
                                        [0, 0, 255],
                                        [156, 156, 156],
                                        [115, 223, 255],
                                        [0, 255, 0],
                                        [255, 255, 0],
                                        [255, 170, 0]])
                    palette = palette * 1.0 / 255
                elif dataID == 3:
                    row = 349
                    col = 1905
                    palette = np.array([[0, 168, 132],
                                        [76, 0, 115],
                                        [0, 0, 0],
                                        [190, 255, 232],
                                        [255, 0, 0],
                                        [115, 0, 0],
                                        [205, 205, 102],
                                        [137, 90, 68],
                                        [215, 158, 158],
                                        [255, 115, 223],
                                        [0, 0, 255],
                                        [156, 156, 156],
                                        [115, 223, 255],
                                        [0, 255, 0],
                                        [255, 255, 0]])
                    palette = palette * 1.0 / 255

                X_result = np.zeros((labels.shape[0], 3))

                for i in range(0, num_class):
                    X_result[np.where(labels == i), 0] = palette[i, 0]
                    X_result[np.where(labels == i), 1] = palette[i, 1]
                    X_result[np.where(labels == i), 2] = palette[i, 2]
                X_result[np.where(gtt == 0), 0] = 1
                X_result[np.where(gtt == 0), 1] = 1
                X_result[np.where(gtt == 0), 2] = 1
                X_result = np.reshape(X_result, (row, col, 3))
                X_mask = np.zeros((row + 2, col + 2, 3))
                X_mask[1:row + 1, 1:col + 1, :] = X_result
                X_result = np.reshape(X_mask, (row + 2, col + 2, 3))
                plt.axis("off")
                plt.imshow(X_result)
                plt.show()
                return X_result
            def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
                real_labels = reallabel_onehot
                we = -torch.mul(real_labels, torch.log(predict))
                we = torch.mul(we, reallabel_mask)
                pool_cross_entropy = torch.sum(we)
                return pool_cross_entropy
            for curr_seed in Seed_List:
                # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
                bestloss = 999
                random.seed(curr_seed)
                train_rand_idx = []

                for i in range(class_count):
                    idx = np.where(gt_reshape == i + 1)[-1]
                    samplesCount = len(idx)
                    real_train_samples_per_class = train_samples_per_class
                    rand_list = [i for i in range(samplesCount)]  # 用于随机的列表

                    if real_train_samples_per_class > 0.5* samplesCount:
                        real_train_samples_per_class =int( 0.5*samplesCount)
                    rand_idx = random.sample(rand_list,
                                             real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
                    rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]

                    train_rand_idx.append(rand_real_idx_per_class_train)

                train_rand_idx = np.array(train_rand_idx)
                train_data_index = []

                for c in range(train_rand_idx.shape[0]):
                    a = train_rand_idx[c]
                    for j in range(a.shape[0]):
                        train_data_index.append(a[j])
                train_data_index = np.array(train_data_index)
                train_data_index = np.array(train_data_index)
                train_data_index = set(train_data_index)
                all_data_index = [i for i in range(len(gt_reshape))]
                allfordraw = all_data_index
                all_data_index = set(all_data_index)

                # 背景像元的标签
                background_idx = np.where(gt_reshape == 0)[-1]
                background_idx = set(background_idx)
                test_data_index = all_data_index - train_data_index - background_idx


                test_data_index = list(test_data_index)
                truetest = test_data_index
                train_data_index = list(train_data_index)

                lentrainsample = len(train_data_index)
                lencontrast = lentrainsample*(lentrainsample-1)
                test = np.array(test_data_index)
                train = np.array(train_data_index)

                allindex = []
                allindex.append(train)
                allindex.append(test)
                train_samples_gt = np.zeros(gt_reshape.shape)

                for i in range(len(train_data_index)):
                    train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
                    pass

                # 获取测试样本的标签图
                test_samples_gt = np.zeros(gt_reshape.shape)
                for i in range(len(test_data_index)):
                    test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
                    pass

                Test_GT = np.reshape(test_samples_gt, [height, width])  # 测试样本图

                # 获取验证集样本的标签图
                def GT_To_One_Hot(gt, class_count):

                    GT_One_Hot = []  # 转化为one-hot形式的标签
                    for i in range(gt.shape[0]):
                        for j in range(gt.shape[1]):
                            temp = np.zeros(class_count, dtype=np.float32)
                            if gt[i, j] != 0:
                                temp[int(gt[i, j]) - 1] = 1
                            GT_One_Hot.append(temp)
                    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
                    return GT_One_Hot

                def cluster_To_One_Hot(gt, class_count):

                    GT_One_Hot = []  # 转化为one-hot形式的标签
                    for i in range(10249):
                        temp = np.zeros(class_count, dtype=np.float32)
                        if gt[i] != 0:
                            temp[int(gt[i] - 1)] = 1
                        GT_One_Hot.append(temp)
                    # GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
                    return GT_One_Hot
                zeros = torch.zeros([height* width]).to(device).float()
                def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                                         printFlag=True):
                    if False == require_AA_KPP:
                        with torch.no_grad():
                            available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                            available_label_count = available_label_idx.sum()  # 有效标签的个数
                            correct_prediction = torch.where(torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                                available_label_idx, zeros).sum()
                            OA = correct_prediction.cpu() / available_label_count
                            return OA
                    else:
                        with torch.no_grad():
                            # 计算OA
                            available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                            available_label_count = available_label_idx.sum()  # 有效标签的个数
                            correct_prediction = torch.where(torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),available_label_idx, zeros).sum()
                            OA = correct_prediction.cpu() / available_label_count
                            OA = OA.cpu().numpy()

                            # 计算AA
                            zero_vector = np.zeros([class_count])
                            output_data = network_output.cpu().numpy()
                            train_samples_gt = train_samples_gt.cpu().numpy()
                            train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

                            output_data = np.reshape(output_data, [height * width, class_count])
                            idx = np.argmax(output_data, axis=-1)
                            for z in range(output_data.shape[0]):
                                if ~(zero_vector == output_data[z]).all():
                                    idx[z] += 1
                            # idx = idx + train_samples_gt
                            count_perclass = np.zeros([class_count])
                            correct_perclass = np.zeros([class_count])
                            for x in range(len(train_samples_gt)):
                                if train_samples_gt[x] != 0:
                                    count_perclass[int(train_samples_gt[x] - 1)] += 1
                                    if train_samples_gt[x] == idx[x]:
                                        correct_perclass[int(train_samples_gt[x] - 1)] += 1
                            test_AC_list = correct_perclass / count_perclass
                            test_AA = np.average(test_AC_list)

                            # 计算KPP
                            test_pre_label_list = []
                            test_real_label_list = []
                            output_data = np.reshape(output_data, [height * width, class_count])
                            idx = np.argmax(output_data, axis=-1)
                            idx = np.reshape(idx, [height, width])
                            for ii in range(height):
                                for jj in range(width):
                                    if Test_GT[ii][jj] != 0:
                                        test_pre_label_list.append(idx[ii][jj] + 1)
                                        test_real_label_list.append(Test_GT[ii][jj])
                            test_pre_label_list = np.array(test_pre_label_list)
                            test_real_label_list = np.array(test_real_label_list)
                            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                              test_real_label_list.astype(np.int16))
                            test_kpp = kappa

                            # 输出
                            if printFlag:
                                print('winsize,窗口大小为：%s'%(winsize))
                                print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                                print('acc per class:')
                                print(test_AC_list)

                            OA_ALL.append(OA)
                            AA_ALL.append(test_AA)
                            KPP_ALL.append(test_kpp)
                            AVG_ALL.append(test_AC_list)

                            # 保存数据信息

                            return OA

                train_samples_gt = np.reshape(train_samples_gt, [height, width])
                test_samples_gt = np.reshape(test_samples_gt, [height, width])


                train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
                test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)


                train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
                test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)

                ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
                # 训练集
                train_label_mask = np.zeros([height * width, class_count])
                temp_ones = np.ones([class_count])
                train_samples_gt = np.reshape(train_samples_gt, [height * width])
                for i in range(height * width):
                    if train_samples_gt[i] != 0:
                        train_label_mask[i] = temp_ones
                train_label_mask = np.reshape(train_label_mask, [height * width, class_count])

                # 测试集
                test_label_mask = np.zeros([height * width, class_count])
                temp_ones = np.ones([class_count])
                test_samples_gt = np.reshape(test_samples_gt, [width * height])
                for i in range(height * width):
                    if test_samples_gt[i] != 0:
                        test_label_mask[i] = temp_ones
                test_label_mask = np.reshape(test_label_mask, [height * width, class_count])

                # 验证集



                ls = LDA_SLIC.LDA_SLIC(orig_data, np.reshape(train_samples_gt, [height, width]), class_count - 1)
                superpixel_scale = Scale
                Q, S, A, Seg = ls.simple_superpixel(scale=superpixel_scale)
                Qsmall, _, Asmall, _ = ls.simple_superpixel(scale=200)

                Q = torch.from_numpy(Q).to(device)
                A = torch.from_numpy(A).to(device)
                Qsmall = torch.from_numpy(Qsmall).to(device)
                Asmall = torch.from_numpy(Asmall).to(device)
                # 转到GPU
                train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
                test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)

                # 转到GPU
                train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
                test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)

                # 转到GPU
                train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
                test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)

                GCNDATA = np.array(GCNdata, np.float32)
                GCNDATA = torch.from_numpy(GCNDATA.astype(np.float32)).to(device)

                channel = t
                Sptransnet = STnet.STnet(bands,channel)
                Sptransnet.to(device)
                sum1 = sum(p.numel() for p in Sptransnet.parameters())

                print("STnet parameters", (sum(p.numel() for p in Sptransnet.parameters())))
                # summary(Sptransnet, (103,9,9))

                net = CNNmodule.CNN(9, 9, channel, class_count)
                sum2 = sum(p.numel() for p in net.parameters())
                print("Siamese CNN parameters", (sum(p.numel() for p in net.parameters())))
                net.to(device)
                GCNnet = DSRGCN.DSGN(height, width, channel, class_count, Q, A, Qsmall, Asmall)
                GCNnet.to(device)
                sum3 = sum(p.numel() for p in GCNnet.parameters())
                print("GCN parameters", (sum(p.numel() for p in GCNnet.parameters())))
                print('Total parameters:',sum1+sum2+sum3)

                SToptimizer = torch.optim.Adam(Sptransnet.parameters(), lr=5e-4, weight_decay=0.00000)
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.00005)
                optimizer1 = torch.optim.Adam(GCNnet.parameters(), lr=5e-4)
                optim_contrast = torch.optim.Adam(net.parameters(), lr=5e-5, weight_decay=0.000)

                print('第 %s 次实验'%str(curr_seed+1))
                # Siamese_input = Sptransnet(GCNDATA,full = True)
                dataloadstart = time.time()
                # winsize = 9
                contrast_data ,classidata = get_data(GCNdata,gt,allindex,winsize,class_count)
                contrast_loader = torch.utils.data.DataLoader(dataset=contrast_data, batch_size=128, shuffle=True,
                                                              drop_last=False)
                labeled_trainloader = torch.utils.data.DataLoader(dataset=classidata, batch_size=16, shuffle=True,
                                                              drop_last=False)
                dataloadingend = time.time() -dataloadstart
                print('data loading time :%.4f'%dataloadingend)

                CNNtrainstart = time.time()
                contrastlearning(80, contrast_loader,labeled_trainloader,optim_contrast,optimizer, SToptimizer,Sptransnet ,net,lentrainsample,lencontrast,GCNnet,GCNDATA,optimizer1, train_samples_gt_onehot, train_label_mask)
                CNNtrainend = time.time() - CNNtrainstart
                print('ALL CNN traing time :%.4f'% CNNtrainend)
                testimestart = time.time()
                GCN_input = Sptransnet(GCNDATA,full=True).detach()
                net.eval()
                output1,_ = net(GCN_input , full=True)
                output1 =  output1.detach()
                GCNtrainstart = time.time()


                alltraintime = time.time()- CNNtrainstart
                Train_Time_ALL.append(alltraintime)
                # print('all train time :%.4f'%alltraintime)
                with torch.no_grad():
                    GCNnet.eval()
                    output,_ = GCNnet(GCN_input, output1, merge = True)

                    del GCN_input,output1,_
                    testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True,
                                                  printFlag=False)
                    testtime = time.time()-testimestart
                    print('test time:',testtime)
                    print('test OA %.2f'%(testOA*10000))
                    print(testOA*1000)
                    classification_map = torch.argmax(output, 1).reshape([height, width]).cpu()
                    entiremap = draw_result(classification_map,gtdraw, dataID=1)
                # classification = output.cpu().detach().numpy()
                # plt.imsave('./result/image/'+str(sampless)+ + 'sample'  + '_OA_' + str('%.2f'(testOA * 10000)) + '.png',
                #            entiremap)
                torch.cuda.empty_cache()
                del Sptransnet,GCNnet,net
                del GCNDATA
            OA_ALL = np.array(OA_ALL)
            AA_ALL = np.array(AA_ALL)
            KPP_ALL = np.array(KPP_ALL)
            AVG_ALL = np.array(AVG_ALL)
            time_cost = np.mean(np.array(Train_Time_ALL))
            OAmean.append(np.mean(OA_ALL))
            print('PCA维度为%s-------------------------'%t)
            print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
            print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
            print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
            print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
            f = open('result/txt/'+str(sampless)+'sample' +'.txt', 'a+')
            OA_results = '\n======================' \
                         + " train num=" + str(train_samples_per_class) \
                         + " CNNlearning rate=" + str(learning_rate) \
                         + 'GCNlearning rate=' + str(5e-3) \
                         + " CNNepochs=" + str(40) \
                         + " GCNepochs=" + str(200) \
                         +"train time cost="+str(time_cost)\
                         + "\n======================" \
                         + "\nOA=" + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                         + "\nAA=" + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                         + '\nkpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                         + '\nacc per class:' + str(np.mean(AVG_ALL, 0)) \
                         + '\n+-' + '\nstd' + str(np.std(AVG_ALL, 0)) + "\n"

            f.write(OA_results)
            f.close()


