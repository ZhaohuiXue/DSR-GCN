%
% DSR-GCN: Differentiated-Scale Restricted Graph Convolutional Network for Few-Shot Hyperspectral Image Classification
%
%    This demo shows the DSR-GCN model for hyperspectral image classification.
%
%    main.py ....... A main script executing experiments under different datasets.
%    dataload.py ....... A script implementing the data loading and preprocessing of CNN branch.
%    PCA_SLIC.py ....... A script implementing the segmentation of hyperspectral image and composition process of graph network.
%    DSRGCN.py ....... A script implementing our main network.
%    CNNmodule.py ....... A script implementing the CNN branch of our main network.
%    STnet.py ....... A script implementing the spectral transform module of our main network.
%   --------------------------------------
%   Note: Required core python libraries
%   --------------------------------------
%   1. python 3.6.12
%   2. pytorch 1.7.1
%   3. torchvision 0.8.2
%   --------------------------------------
%   Cite:
%   --------------------------------------
%
%   [1] Z. Xue and Z. Liu, "DSR-GCN: Differentiated-Scale Restricted Graph Convolutional Network for Few-Shot Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-18, 2023, Art no. 5504918, doi: 10.1109/TGRS.2023.3253248.
%   --------------------------------------
%   Copyright & Disclaimer
%   --------------------------------------
%
%   The programs contained in this package are granted free of charge for
%   research and education purposes only. 
%
%   Copyright (c) 2023 by Zhaohui Xue & Zhiwei Liu
%   zhaohui.xue@hhu.edu.cn & lzw_hhu@163.com
%   --------------------------------------
%   For full package:
%   --------------------------------------
%   https://sites.google.com/site/zhaohuixuers/
