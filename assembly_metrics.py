import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt

from scipy.stats import pearsonr,spearmanr

import torch
import argparse
import utils
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import os


# 计算数据的RDM
def compute_rdm(data_matrix):
    # 计算数据矩阵中每对样本之间的欧氏距离
    distances = pdist(data_matrix, 'cosine')
    
    # 将距离转换成RDM（代表性差异矩阵）
    rdm = squareform(distances)
    
    return rdm


def vis_rdm(rdm_matrix):
    # 可视化RDM
    plt.imshow(rdm_matrix, cmap='Greys', origin='upper')
    plt.colorbar(label='Dissimilarity')
    plt.title('Representational Dissimilarity Matrix')
    plt.xlabel('Samples')
    plt.ylabel('Samples')
    plt.savefig('./pics/rdm.jpg')



def compare_rdm(rdm_matrix1,rdm_matrix2):
    # 将两个RDMs转换为成对距离矩阵
    rdm1 = squareform(rdm_matrix1)
    rdm2 = squareform(rdm_matrix2)

    # 计算Pearson相关系数和p值
    corr_pearson, p_value = pearsonr(rdm1, rdm2)

    ###print("Pearson Correlation:{:.4f}".format(corr_pearson))
    # print("P-value:", p_value)

    corr_spear, p_value = spearmanr(rdm1, rdm2)
    ###print("Spearman Correlation:{:.4f}".format(corr_spear))
    # print("P-value:", p_value)

    return corr_pearson, corr_spear

def infer(encoder, dataset,args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    embeddings_list = []
    for i, data in enumerate(data_loader):
            data = data.to(args.device)
            embeddings = encoder(data).detach().cpu().numpy()
            embeddings_list.append(embeddings)
    embeddings = np.concatenate(tuple(embeddings_list), axis=0)
    return embeddings

def compute_double_rdm(encoder_1,encoder_2,dataset,args):
    embeddings_1 = infer(encoder_1, dataset,args)
    embeddings_2 = infer(encoder_2, dataset,args)

    rdm_matrix_1 = compute_rdm(embeddings_1)
    rdm_matrix_2 = compute_rdm(embeddings_2)


    corr_pearson, corr_spear = compare_rdm(rdm_matrix_1,rdm_matrix_2)

    return corr_pearson, corr_spear

def compute_triple_rdm(encoder_1,encoder_2,encoder_3,dataset,args):
    embeddings_1 = infer(encoder_1, dataset,args)
    embeddings_2 = infer(encoder_2, dataset,args)
    embeddings_3 = infer(encoder_3, dataset,args)

    rdm_matrix_1 = compute_rdm(embeddings_1)
    rdm_matrix_2 = compute_rdm(embeddings_2)
    rdm_matrix_3 = compute_rdm(embeddings_3)

    corr_pearson_12, corr_spear_12 = compare_rdm(rdm_matrix_1,rdm_matrix_2)
    corr_pearson_13, corr_spear_13 = compare_rdm(rdm_matrix_1,rdm_matrix_3)
    corr_pearson_23, corr_spear_23 = compare_rdm(rdm_matrix_2,rdm_matrix_3)

    return (corr_pearson_12+corr_pearson_13+corr_pearson_23)/3, (corr_spear_12+corr_spear_13+corr_spear_23)/3 \
        , corr_spear_12, corr_spear_13, corr_spear_23
'''
# 创建一个虚拟的神经数据矩阵，每一行代表不同的样本或刺激条件，每一列代表不同的神经单元或特征
neural_data = np.random.rand(10, 20)  # 假设有10个样本和20个特征
# 计算RDM
rdm_matrix = compute_rdm(neural_data)
'''
