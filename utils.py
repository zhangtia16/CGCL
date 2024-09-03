# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import random
import torch
import torch.nn as nn
import pickle
import os
from sklearn.model_selection import StratifiedKFold




def contrastive_loss_calculate(temperature, model_q_hidden_feats, model_k_hidden_feats, memory_bank):
    '''
    hidden feats must be normalized.
    '''
    
    q = nn.functional.normalize(model_q_hidden_feats, dim=1)
    k = nn.functional.normalize(model_k_hidden_feats, dim=1)
    loss = 0
    # print('max value', torch.max(torch.mm(q, k.transpose(0,1))))
    l_pos = torch.diag(torch.exp(torch.mm(q, k.transpose(0,1))/temperature), 0)
    #l_neg = torch.sum(torch.exp(torch.mm(q, memory_bank.transpose(0,1))/temperature), dim=1)
    l_neg = torch.sum(torch.exp(torch.mm(q, k.transpose(0,1))/temperature), dim=1)
    #print('two part size',l_pos.size(), l_neg.size())
    loss += torch.sum(-1.0*torch.log(l_pos/l_neg))
    return loss


def dual_temperature_contrastive_loss_calculate(positive_temperature, negative_temperature, model_q_hidden_feats, model_k_hidden_feats):
    '''
    hidden feats must be normalized.
    '''
    
    q = nn.functional.normalize(model_q_hidden_feats, dim=1)
    k = nn.functional.normalize(model_k_hidden_feats, dim=1)
    loss = 0
    # print('max value', torch.max(torch.mm(q, k.transpose(0,1))))
    l_pos = torch.diag(torch.exp(torch.mm(q, k.transpose(0,1))/positive_temperature), 0)
    #l_neg = torch.sum(torch.exp(torch.mm(q, memory_bank.transpose(0,1))/temperature), dim=1)
    
    neg_matrix = torch.mm(q, k.transpose(0,1))/negative_temperature
    neg_diag = torch.diag(neg_matrix, 0)
    neg_matrix = neg_matrix - torch.diag(neg_diag)
    #print(neg_matrix)
    l_neg = torch.sum(torch.exp(neg_matrix), dim=1)
    #print('two part size',l_pos.size(), l_neg.size())
    loss += torch.sum(-1.0*torch.log(l_pos/(l_neg+l_pos)))
    #loss += torch.sum(-1.0*torch.log(l_pos/l_neg))
    return loss


def contrastive_loss_calculate_with_memory(device, temperature, sample_number, query_index, model_q_hidden_feats, model_k_hidden_feats, memory_bank):
    '''
    hidden feats must be normalized.
    '''
    
    q = nn.functional.normalize(model_q_hidden_feats, dim=1)
    k = nn.functional.normalize(model_k_hidden_feats, dim=1)
    loss = 0
    # print('max value', torch.max(torch.mm(q, k.transpose(0,1))))
    l_pos = torch.diag(torch.exp(torch.mm(q, k.transpose(0,1))/temperature), 0)
    neg_matrix = torch.mm(q, memory_bank.transpose(0,1)/temperature)
    mask_number = memory_bank.shape[0] - sample_number
    indices = []
    for i in range(neg_matrix.shape[0]):
        mask_vector = torch.zeros(1,memory_bank.shape[0])
        index_rand = torch.randperm(memory_bank.shape[0])
        index = index_rand[0:mask_number].numpy()
        if query_index[i] in index:
            index.remove(query_index[i])
            index.append(index_rand[mask_number])
        index = torch.Tensor(index).type('torch.LongTensor')
        mask_vector = mask_vector.index_fill(1,index,1)
        indices.append(mask_vector)
    masked_matrix = torch.Tensor(np.concatenate(tuple(indices), axis=0)).bool().to(device)
    neg_matrix = neg_matrix.masked_fill(mask = masked_matrix, value=0)
    #print(neg_matrix)
    l_neg = torch.sum(torch.exp(neg_matrix), dim=1)
    #print('two part size',l_pos.size(), l_neg.size())
    loss += torch.sum(-1.0*torch.log(l_pos/(l_neg+l_pos)))
    return loss

def contrastive_tri_loss_calculate(temperature, model_q_hidden_feats, model_k1_hidden_feats, model_k2_hidden_feats, memory_bank):
    '''
    hidden feats must be normalized.
    '''
    
    q = nn.functional.normalize(model_q_hidden_feats, dim=1)
    k_1 = nn.functional.normalize(model_k1_hidden_feats, dim=1)
    k_2 = nn.functional.normalize(model_k2_hidden_feats, dim=1) 
    loss = 0
    # print('max value', torch.max(torch.mm(q, k.transpose(0,1))))
    l_pos_k1 = torch.diag(torch.exp(torch.mm(q, k_1.transpose(0,1))/temperature), 0)
    #l_neg = torch.sum(torch.exp(torch.mm(q, memory_bank.transpose(0,1))/temperature), dim=1)
    l_neg_k1 = torch.sum(torch.exp(torch.mm(q, k_1.transpose(0,1))/temperature), dim=1)
    loss_k1 = torch.sum(-1.0*torch.log(l_pos_k1/l_neg_k1))
    #print('two part size',l_pos.size(), l_neg.size())
    l_pos_k2 = torch.diag(torch.exp(torch.mm(q, k_2.transpose(0,1))/temperature), 0) 
    l_neg_k2 = torch.sum(torch.exp(torch.mm(q, k_2.transpose(0,1))/temperature), dim=1) 
    loss_k2 = torch.sum(-1.0*torch.log(l_pos_k2/l_neg_k2))
    
    loss = (loss_k1 + loss_k2)/2
    
    return loss


def set_random_seed(seed):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    #dgl.random.seed(seed)


class Early_Stopper:
    def __init__(self, patience, save_paths, min_epoch=-1):
        self.patience = patience
        self.bad_counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.save_paths = save_paths
        self.min_epoch = min_epoch

    def step(self, score, epoch, encoders):
        if epoch < self.min_epoch:
            return self.early_stop
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(encoders)
        elif score < self.best_score:
            self.bad_counter += 1         
            #print(f'EarlyStopping bad_counter: {self.bad_counter} out of {self.patience}')
            if self.bad_counter >= self.patience:
                self.early_stop = True
                print()
                ###print(f'EarlyStop at Epoch {epoch} with patience {self.patience}')
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(encoders)
            self.bad_counter = 0
        return self.early_stop

    def save_checkpoint(self, encoders):
        '''Saves model when validation loss decrease.'''
        for encoder_idx in range(len(encoders)):
            torch.save(encoders[encoder_idx].state_dict(), self.save_paths[encoder_idx])