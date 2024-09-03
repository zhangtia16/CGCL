
import argparse
import glob
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from models.model_ft import model_ft
from models.GIN_ft import GIN_ft
from models.GAT import GAT
from models.GCN import GCN
from models.Set2SetNet import Set2SetNet
from models.SortPool import SortPool
from models.GIN import GIN
from models.HGP_SL import HGP_SL
from models.LogReg import LogReg
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from memory_bank import dequeue_and_enqueue
from utils import contrastive_loss_calculate, dual_temperature_contrastive_loss_calculate, contrastive_tri_loss_calculate
from torch_geometric.transforms import OneHotDegree, Constant
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import warnings

import utils
import assembly_metrics

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=888, help='random seed')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.005, help='model k learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='MUTAG', help='DD/PROTEINS/NCI1')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=10000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')
parser.add_argument('--temperature', type=float, default=0.07, help='The temperature for contrastive loss.')
parser.add_argument('--node_attr', type=str, default='default', help='node has the attribute')  # 有时间写成字典，不用一个个查了


parser.add_argument('--encoder_1', type=str, default='gin', help='encoder 1 of CGCL')
parser.add_argument('--encoder_2', type=str, default='gin', help='encoder 2 of CGCL')
parser.add_argument('--encoder_3', type=str, default='gin', help='encoder 3 of CGCL')
parser.add_argument('--save_root', type=str, default='./checkpoints', help='encoder 2 of CGCL')
parser.add_argument('--repeat_times', type=int, default=10, help='number of repeating experiments')


args = parser.parse_args()
print(args)
utils.set_random_seed(args.seed)

if args.node_attr == 'default':
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)
elif args.node_attr == 'constant':
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, transform = Constant())  
elif args.node_attr == 'onehot':
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, transform = OneHotDegree(200))  

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features


num_training = int(len(dataset) * 0.1)
num_val = int(len(dataset) * 0)
num_test = len(dataset) - (num_training + num_val)

assembly = '{}-{}-{}'.format(args.encoder_1,args.encoder_2,args.encoder_3)
save_path_1 = os.path.join(args.save_root, args.dataset,'{}_encoder1_{}.pth'.format(assembly, args.encoder_1))
save_path_2 = os.path.join(args.save_root, args.dataset,'{}_encoder2_{}.pth'.format(assembly, args.encoder_2))
save_path_3 = os.path.join(args.save_root, args.dataset,'{}_encoder3_{}.pth'.format(assembly, args.encoder_3))


def train_three_models(encoder_1, encoder_2, encoder_3, optimizer_1, optimizer_2, optimizer_3, encoder_1_mb, encoder_2_mb, encoder_3_mb, train_loader):
    min_loss = 1e10
    patience_cnt = 0
    best_epoch = 0

    t = time.time()
    encoder_1.train()
    encoder_2.train()
    encoder_3.train()

    encoder_1_losses, encoder_2_losses, encoder_3_losses, total_losses = [], [], [], []
    
    stopper = utils.Early_Stopper(patience=args.patience, save_paths=[save_path_1, save_path_2, save_path_3], min_epoch=-1)

    for epoch in range(args.epochs):
        total_loss = 0.0
        encoder_1_loss = 0.0
        encoder_2_loss = 0.0
        encoder_3_loss = 0.0
        start_point = 0
        
        for i, data in enumerate(train_loader):
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()
            #optimizer.zero_grad()
            data = data.to(args.device)
            encoder_1_learned_representations = encoder_1(data)
            encoder_2_learned_representations = encoder_2(data)
            encoder_3_learned_representations = encoder_3(data) 
            #print(model_k_learned_representations.shape)
            #print(model_q_learned_representations.shape)
            # update memory bank
            query_index = torch.arange(start_point, start_point+len(data.y))
            encoder_1_mb = dequeue_and_enqueue(encoder_1_learned_representations.detach().clone(), query_index, encoder_1_mb)
            encoder_2_mb = dequeue_and_enqueue(encoder_2_learned_representations.detach().clone(), query_index, encoder_2_mb)
            encoder_3_mb = dequeue_and_enqueue(encoder_3_learned_representations.detach().clone(), query_index, encoder_3_mb)
            start_point += len(data.y)
            # compute contrastive loss
            loss_1 = contrastive_tri_loss_calculate(args.temperature, encoder_1_learned_representations, encoder_2_learned_representations.detach().clone(), encoder_3_learned_representations.detach().clone(), encoder_1_mb)
            loss_2 = contrastive_tri_loss_calculate(args.temperature, encoder_2_learned_representations, encoder_1_learned_representations.detach().clone(), encoder_3_learned_representations.detach().clone(), encoder_2_mb)
            loss_3 = contrastive_tri_loss_calculate(args.temperature, encoder_3_learned_representations, encoder_2_learned_representations.detach().clone(), encoder_1_learned_representations.detach().clone(), encoder_3_mb)
            #loss = model_k_loss + model_q_loss
            encoder_1_loss += loss_1.item()
            encoder_2_loss += loss_2.item()
            encoder_3_loss += loss_3.item()
  
            loss_1.backward()
            optimizer_1.step()
            loss_2.backward()
            optimizer_2.step()
            loss_3.backward()
            optimizer_3.step()

            

        total_loss = encoder_1_loss + encoder_2_loss+ encoder_3_loss
        total_losses.append(total_loss)
        encoder_1_losses.append(encoder_1_loss)
        encoder_2_losses.append(encoder_2_loss)
        encoder_3_losses.append(encoder_3_loss)
        # print('Epoch: {:04d}'.format(epoch), 'encoder_1_loss: {:.2f}'.format(encoder_1_loss), 'encoder_2_loss: {:.2f}'.format(encoder_2_loss), 'encoder_3_loss: {:.6f}'.format(encoder_3_loss),
        #       'total_loss: {:.2f}'.format(total_loss), 'time: {:.2f}s'.format(time.time() - t))
        
        # early stop
        encoders = [encoder_1, encoder_2, encoder_3]
        stop = stopper.step(-total_loss, epoch, encoders) 
        if stop:
            ###print('best epoch :', stopper.best_epoch)
            break

    ###print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch,  encoder_1_losses, encoder_2_losses, encoder_3_losses, total_losses


def infer(trained_model, dataset):
    pass
    
def compute_test_svm(trained_model, dataset, search):
    trained_model.eval()
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    accuracies = []
    embeddings_list = []
    labels_list = [] 
    for i, data in enumerate(data_loader):
            data = data.to(args.device)
            embeddings = trained_model(data).detach().cpu().numpy()
            labels = data.y.cpu().numpy()
            embeddings_list.append(embeddings)
            labels_list.append(labels)
    embeddings = np.concatenate(tuple(embeddings_list), axis=0)
    labels = np.concatenate(tuple(labels_list), axis=0)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=888)
    if search:
            params = {"C": [1, 10, 100, 1000, 10000, 100000]}
            classifier = GridSearchCV(
                SVC(), params, cv=kf, scoring="accuracy", verbose=0, n_jobs=-1
            )
            grid_result = classifier.fit(embeddings, labels)
            ###print("Best: %f using %s" % (grid_result.best_score_,classifier.best_params_))
            return grid_result.best_score_
    else:
        for train_index, test_index in kf.split(embeddings, labels):
            x_train, x_test = embeddings[train_index], embeddings[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            classifier = SVC(C=1000)
            classifier.fit(x_train, y_train)
            accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
            return {"Micro-F1": np.mean(accuracies), "std": np.std(accuracies)}


def finetune_model(pretrained_model, dataset, ft_epoch):
     # prepare data
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    bad_counter = 0
    best = 10000
    loss_values = []
    best_epoch = 0
    patience = 1000
    finetune_epoch = ft_epoch 
    t = time.time()

    xent = nn.CrossEntropyLoss()
    model = model_ft(args.nhid, dataset.num_classes, pretrained_model)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    best_acc = torch.zeros(1)
    if torch.cuda.is_available():
        print('Using CUDA')
        model.to(args.device)
        best_acc = best_acc.to(args.device)
    for epoch in range(finetune_epoch):
        model.train()
        opt.zero_grad()
        val_loss = 0.0
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            logits = model(data)
            loss= xent(logits, data.y)
            loss.backward()
            opt.step()
        for i, data in enumerate(val_loader):
            data = data.to(args.device)
            logits = model(data)
            val_loss += xent(logits, data.y).item()
        loss_values.append(val_loss)    
        
        
        torch.save(model.state_dict(), '{}.ft.{}.pkl'.format(epoch,args.device))   
        if loss_values[-1] < best:
           best = loss_values[-1]
           best_epoch = epoch
           bad_counter = 0
        else:
           bad_counter += 1
        
        if bad_counter == patience:
            break

        # test
        correct = 0
        for i, data in enumerate(test_loader):
            data = data.to(args.device)
            logits = model(data)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(data.y).sum().item()
        acc = correct/num_test
        
        files = glob.glob('*.ft.{}.pkl'.format(args.device))
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
               os.remove(file)    

        print('Epoch: {:04d} val_loss: {:.6f} time: {:.6f}s test_acc:{:.6f}'.format(epoch,val_loss,time.time() - t,acc)) 
    # files = glob.glob('*.ft.{}.pkl'.format(args.device))
    
    # for file in files:
    #     epoch_nb = int(file.split('.')[0])
    #     if epoch_nb > best_epoch:
    #         os.remove(file)
    
    # print("Optimization Finished!")         
    # print('Loading {}th epoch'.format(best_epoch))
    # model.load_state_dict(torch.load('{}.ft.{}.pkl'.format(best_epoch,args.device)))
    
    # files = glob.glob('*.ft.{}.pkl'.format(args.device))
    # for file in files:
    #         os.remove(file)

    correct = 0.0
    for i, data in enumerate(test_loader):
            data = data.to(args.device)
            logits = model(data)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(data.y).sum().item()
    acc = correct/num_test
    return model.pretrain_model, acc


# VIS
def vis(best_epoch = 179):
    encoder_1 = GIN(args).to(args.device) 
    encoder_1.load_state_dict(torch.load('{}.1.{}.pth'.format(best_epoch, args.device)))
    print('loading done!')
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    embeddings_list = []
    labels_list = [] 
    for i, data in enumerate(data_loader):
        data = data.to(args.device)
        embeddings = encoder_1(data).detach().cpu().numpy()
        labels = data.y.cpu().numpy()
        embeddings_list.append(embeddings)
        labels_list.append(labels)         

    embeddings = np.concatenate(tuple(embeddings_list), axis=0)
    labels = np.concatenate(tuple(labels_list), axis=0)
    print(embeddings.shape)
    print(labels.shape)
    print(set(labels))

    embeddings_tsne = TSNE(n_components=2).fit_transform(embeddings)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 20, labels[:])
    plt.savefig("./pics/{}_gin.pdf".format(args.dataset),dpi=50)



def init_encoders(args):
    if args.encoder_1 == 'gin':
        encoder_1 = GIN(args).to(args.device)
    elif args.encoder_1 == 'gat':
        encoder_1 = GAT(args).to(args.device)
    elif args.encoder_1 == 'gcn':
        encoder_1 = GCN(args).to(args.device)
    elif args.encoder_1 == 'set':
        encoder_1 = Set2SetNet(args).to(args.device)

    if args.encoder_2 == 'gin':
        encoder_2 = GIN(args).to(args.device)
    elif args.encoder_2 == 'gat':
        encoder_2 = GAT(args).to(args.device)
    elif args.encoder_2 == 'gcn':
        encoder_2 = GCN(args).to(args.device)
    elif args.encoder_2 == 'set':
        encoder_2 = Set2SetNet(args).to(args.device)
    
    if args.encoder_3 == 'gin':
        encoder_3 = GIN(args).to(args.device)
    elif args.encoder_3 == 'gat':
        encoder_3 = GAT(args).to(args.device)
    elif args.encoder_3 == 'gcn':
        encoder_3 = GCN(args).to(args.device)
    elif args.encoder_3 == 'set':
        encoder_3 = Set2SetNet(args).to(args.device)

    return encoder_1, encoder_2, encoder_3

if __name__ == '__main__':
        encoder_2_acc_collec , encoder_1_acc_collec, encoder_3_acc_collec =[], [], []
        corr_pearson_collec, corr_spear_collec = [], []
        corr_spear_collec_12, corr_spear_collec_13, corr_spear_collec_23 = [], [], []
        stop_loss_collec = []

        for repeat_idx in range(args.repeat_times):
            ###print('Repeation:{}'.format(repeat_idx))

            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            encoder_1, encoder_2, encoder_3 = init_encoders(args)
            
            optimizer_1 = torch.optim.Adam(encoder_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_2 = torch.optim.Adam(encoder_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)        
            optimizer_3 = torch.optim.Adam(encoder_3.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # Model training
            stdv = 1.0 / math.sqrt(args.num_features / 3)
            encoder_1_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            encoder_2_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)     
            encoder_3_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)

            best_epoch,  model_1_losses, model_2_losses, model_3_losses,  total_losses = \
                train_three_models(encoder_1, encoder_2, encoder_3, optimizer_1, optimizer_2, optimizer_3, encoder_1_mb, encoder_2_mb, encoder_3_mb, data_loader)
            
            ###print('Loading trained encoder 1 from {}'.format(save_path_1))
            ###print('Loading trained encoder 2 from {}'.format(save_path_2))
            ###print('Loading trained encoder 3 from {}'.format(save_path_3))

            encoder_1.load_state_dict(torch.load(save_path_1))
            encoder_2.load_state_dict(torch.load(save_path_2))
            encoder_3.load_state_dict(torch.load(save_path_3))

            encoder_1_acc = compute_test_svm(encoder_1,dataset,search=True)
            encoder_2_acc = compute_test_svm(encoder_2,dataset,search=True)
            encoder_3_acc = compute_test_svm(encoder_3,dataset,search=True)

            encoder_1_acc_collec.append(encoder_1_acc)
            encoder_2_acc_collec.append(encoder_2_acc)
            encoder_3_acc_collec.append(encoder_3_acc)

            ###print('encoder_1 acc: {:.4f}'.format(encoder_1_acc))
            ###print('encoder_2 acc: {:.4f}'.format(encoder_2_acc))
            ###print('encoder_3 acc: {:.4f}'.format(encoder_3_acc))


            # assembly metric
            corr_pearson, corr_spear, corr_spear_12, corr_spear_13, corr_spear_23 = \
                assembly_metrics.compute_triple_rdm(encoder_1,encoder_2,encoder_3,dataset,args)

            corr_pearson_collec.append(corr_pearson)
            corr_spear_collec.append(corr_spear)
            corr_spear_collec_12.append(corr_spear_12)
            corr_spear_collec_13.append(corr_spear_13)
            corr_spear_collec_23.append(corr_spear_23)

            stop_loss_collec.append(total_losses[-1]/(3*len(dataset)))

            
            
        print('---------------------------------------------------------------------')
        print('Dataset:{} | assembly:{} | encoder_1:{} | encoder_2:{} | encoder_3:{}'.format(args.dataset,assembly,args.encoder_1,args.encoder_2,args.encoder_3))
        print('Average encoder_1 acc: {:.4f}±{:.4f}'.format(np.mean(encoder_1_acc_collec), np.std(encoder_1_acc_collec)))
        print('Average encoder_2 acc: {:.4f}±{:.4f}'.format(np.mean(encoder_2_acc_collec), np.std(encoder_2_acc_collec)))
        print('Average encoder_3 acc: {:.4f}±{:.4f}'.format(np.mean(encoder_3_acc_collec), np.std(encoder_3_acc_collec)))
        print('---------Assembly Metrics---------')
        print('pearson rdm: {:.4f}±{:.4f}'.format(np.mean(corr_pearson_collec), np.std(corr_pearson_collec)))
        print('spear rdm: {:.4f}±{:.4f}'.format(np.mean(corr_spear_collec), np.std(corr_spear_collec)))
        print('stop loss: {:.4f}±{:.4f}'.format(np.mean(stop_loss_collec), np.std(stop_loss_collec)))
        print('---------Rdm Details---------')
        print('encoder_12 spear rdm: {:.4f}±{:.4f}'.format(np.mean(corr_spear_collec_12), np.std(corr_spear_collec_12)))
        print('encoder_13 spear rdm: {:.4f}±{:.4f}'.format(np.mean(corr_spear_collec_13), np.std(corr_spear_collec_13)))
        print('encoder_23 spear rdm: {:.4f}±{:.4f}'.format(np.mean(corr_spear_collec_23), np.std(corr_spear_collec_23)))
