# -*- coding: utf-8 -*-

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
from memory_bank import dequeue_and_enqueue, update_whole_memory_bank
from utils import contrastive_loss_calculate, dual_temperature_contrastive_loss_calculate, contrastive_loss_calculate_with_memory
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
parser.add_argument('--lr_q', type=float, default=0.001, help='model q learning rate')
parser.add_argument('--lr_k', type=float, default=0.001, help='model k learning rate')
parser.add_argument('--lr', type=float, default=0.05, help='model k learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
#parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1')
parser.add_argument('--device', type=str, default='cuda:1', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=10000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')
parser.add_argument('--loss_weight', type=float, default=0.1, help='The weight of contrastive loss term.')
parser.add_argument('--temperature', type=float, default=0.1, help='The temperature for contrastive loss.')
parser.add_argument('--positive_temperature', type=float, default=0.07, help='The temperature for contrastive loss.')
parser.add_argument('--negative_temperature', type=float, default=0.8, help='The temperature for contrastive loss.')
parser.add_argument('--node_attr', type=str, default='default', help='node has the attribute')
parser.add_argument('--sample_number', type=int, default=128, help='number of sampled negative graph')


parser.add_argument('--encoder_1', type=str, default='gin', help='encoder 1 of CGCL')
parser.add_argument('--encoder_2', type=str, default='gin', help='encoder 2 of CGCL')
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
    # Constant()
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset) * 0.1)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)


assembly = '{}-{}'.format(args.encoder_1,args.encoder_2)
save_path_1 = os.path.join(args.save_root, args.dataset,'{}_encoder1_{}.pth'.format(assembly, args.encoder_1))
save_path_2 = os.path.join(args.save_root, args.dataset,'{}_encoder2_{}.pth'.format(assembly, args.encoder_2))

def train_two_models(encoder_1, encoder_2, optimizer_1, optimizer_2, model_1_mb, model_2_mb, train_loader):
    """
    Parameters
    ----------
        encoder_1/k : 
        optimizer_1/k: 
        encoder_1/k_memory_bank: 
        train_loader: Dataloader(...)

    Outputs
    ------
        best_epoch :
        encoder_1/k_contrastive_loss :
        total_losses :
    
    Saved files
    ------
        {}.q.{}.pth : trained model with format(epoch, 'cuda:x')
        {}.k.{}.pth :
    """

    # init
    min_loss = 1e10
    patience_cnt = 0
    best_epoch = 0
    t = time.time()
    encoder_1.train()
    encoder_2.train()
    model_1_losses, model_2_losses, total_losses = [], [], []

    
    stopper = utils.Early_Stopper(patience=args.patience, save_paths=[save_path_1, save_path_2], min_epoch=-1)

    
    for epoch in range(args.epochs):
        total_loss = 0.0
        model_2_loss = 0.0
        model_1_loss = 0.0
        start_point = 0
        
        # train
        for i, data in enumerate(train_loader):
            optimizer_2.zero_grad()
            optimizer_1.zero_grad()

            data = data.to(args.device)
            model_1_learned_representations = encoder_1(data)
            model_2_learned_representations = encoder_2(data)
            

            # update memory bank
            query_index = torch.arange(start_point, start_point+len(data.y))
            model_1_mb = dequeue_and_enqueue(model_1_learned_representations.detach().clone(), query_index, model_1_mb)
            model_2_mb = dequeue_and_enqueue(model_2_learned_representations.detach().clone(), query_index, model_2_mb)
            
            start_point += len(data.y)
            # compute  contrastive loss
            loss_1 = contrastive_loss_calculate(args.temperature, model_1_learned_representations, model_2_learned_representations.detach().clone(), model_2_mb)
            loss_2 = contrastive_loss_calculate(args.temperature, model_2_learned_representations, model_1_learned_representations.detach().clone(), model_1_mb)
            
            #loss = model_2_loss + model_1_loss
            model_1_loss += loss_1.item()
            model_2_loss += loss_2.item()

            loss_1.backward()
            optimizer_1.step()
            loss_2.backward()
            optimizer_2.step()
            
            

        total_loss = model_1_loss + model_2_loss
        total_losses.append(total_loss)
        model_1_losses.append(model_1_loss)
        model_2_losses.append(model_2_loss)
        #print('Epoch: {:04d}'.format(epoch), 'model_2_loss: {:.4f}'.format(model_2_loss), 'model_1_loss: {:.4f}'.format(model_1_loss),'total_loss: {:.4f}'.format(total_loss), 'time: {:.4f}s'.format(time.time() - t))
        

        # early stop
        encoders = [encoder_1, encoder_2]
        stop = stopper.step(-total_loss, epoch, encoders) 
        if stop:
            ###print('best epoch :', stopper.best_epoch)
            break


    ###print('Optimization Finished! Total time elapsed: {:.4f}'.format(time.time() - t))
    return best_epoch,  model_1_losses, model_2_losses, total_losses



def train_two_models_with_memery(encoder_1, encoder_2, optimizer_2, optimizer_1, model_2_mb, model_1_mb, train_loader):
    min_loss = 1e10
    patience_cnt = 0
    best_epoch = 0
    t = time.time()
    encoder_1.train()
    encoder_2.train()

    #val_acc, train_acc, train_celoss, train_loss= [],[],[],[]
    model_1_losses, model_2_losses, total_losses = [], [], []
    for epoch in range(args.epochs):
        print('epoch: {:04d}'.format(epoch))
        total_loss = 0.0
        model_2_loss = 0.0
        model_1_loss = 0.0

        start_point = 0
        
        for i, data in enumerate(train_loader):
            optimizer_2.zero_grad()
            optimizer_1.zero_grad()
            #optimizer.zero_grad()
            data = data.to(args.device)
            model_2_learned_representations = encoder_2(data)
            model_1_learned_representations = encoder_1(data)
            #print(model_2_learned_representations.shape)
            #print(model_1_learned_representations.shape)
            # update encoder_2 memory bank
            query_index = torch.arange(start_point, start_point+len(data.y))
            # model_2_mb = dequeue_and_enqueue(model_2_learned_representations.detach().clone(), query_index, model_2_mb)
            model_2_mb = update_whole_memory_bank(args, encoder_2, model_2_mb, train_loader)
            # update encoder_1 memory bank
            model_1_mb = update_whole_memory_bank(args, encoder_1, model_1_mb, train_loader)
            #model_1_mb = dequeue_and_enqueue(model_1_learned_representations.detach().clone(), query_index, model_1_mb)
            start_point += len(data.y)
            # compute encoder_2 contrastive loss
            model_2_loss = contrastive_loss_calculate_with_memory(args.device, args.temperature, args.sample_number, query_index, model_2_learned_representations, model_1_learned_representations.detach().clone(), model_1_mb)
            # compute encoder_1 contrastive loss
            model_1_loss = contrastive_loss_calculate_with_memory(args.device, args.temperature, args.sample_number, query_index, model_1_learned_representations, model_2_learned_representations.detach().clone(), model_2_mb)
            #loss = model_2_loss + model_1_loss
            model_2_loss += model_2_loss.item()
            model_1_loss += model_1_loss.item()
            #loss.backward()
            model_2_loss.backward()
            optimizer_2.step()
            model_1_loss.backward()
            optimizer_1.step()
            #loss.backward()
            #optimizer.step()
            

        total_loss = model_2_loss + model_1_loss 
        total_losses.append(total_loss)
        model_1_losses.append(model_1_loss)
        model_2_losses.append(model_2_loss)
        print('Epoch: {:04d}'.format(epoch), 'model_2_loss: {:.4f}'.format(model_2_loss), 'model_1_loss: {:.4f}'.format(model_1_loss),
              'total_loss: {:.4f}'.format(total_loss), 'time: {:.4f}s'.format(time.time() - t))
        torch.save(encoder_2.state_dict(), '{}.k.{}.pth'.format(epoch, args.device))
        torch.save(encoder_1.state_dict(), '{}.q.{}.pth'.format(epoch, args.device))
        if total_losses[-1] < min_loss:
            min_loss = total_losses[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
        
        
        if patience_cnt == args.patience:
            break
        
        
        files_k = glob.glob('*.k.{}.pth'.format(args.device))
        for f in files_k:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)
                print("remove")
        files_q = glob.glob('*.q.{}.pth'.format(args.device))
        for f in files_q:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files_k = glob.glob('*.k.{}.pth'.format(args.device))
    for f in files_k:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    files_q = glob.glob('*.q.{}.pth'.format(args.device))
    for f in files_q:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.4f}'.format(time.time() - t))

    return best_epoch,  model_1_losses, model_2_losses, total_losses

def train_two_models_dual_temperature(encoder_1, encoder_2, optimizer_2, optimizer_1, train_loader):
    min_loss = 1e10
    patience_cnt = 0
    best_epoch = 0

    t = time.time()
    encoder_1.train()
    encoder_2.train()

    #val_acc, train_acc, train_celoss, train_loss= [],[],[],[]
    model_1_losses, model_2_losses, total_losses = [], [], []
    for epoch in range(args.epochs):
        print('epoch: {:04d}'.format(epoch))
        total_loss = 0.0
        model_2_loss = 0.0
        model_1_loss = 0.0

        start_point = 0
        
        for i, data in enumerate(train_loader):
            optimizer_2.zero_grad()
            optimizer_1.zero_grad()
            #optimizer.zero_grad()
            data = data.to(args.device)
            model_2_learned_representations = encoder_2(data)
            model_1_learned_representations = encoder_1(data)
            #print(model_2_learned_representations.shape)
            #print(model_1_learned_representations.shape)
            # compute encoder_2 contrastive loss
            model_2_loss = dual_temperature_contrastive_loss_calculate(args.positive_temperature, args.negative_temperature, model_2_learned_representations, model_1_learned_representations.detach().clone())
            # compute encoder_1 contrastive loss
            model_1_loss = dual_temperature_contrastive_loss_calculate(args.positive_temperature, args.negative_temperature, model_1_learned_representations, model_2_learned_representations.detach().clone())
            #loss = model_2_loss + model_1_loss
            model_2_loss += model_2_loss.item()
            model_1_loss += model_1_loss.item()
            #loss.backward()
            model_2_loss.backward()
            optimizer_2.step()
            model_1_loss.backward()
            optimizer_1.step()
            #loss.backward()
            #optimizer.step()
            

        total_loss = model_2_loss + model_1_loss 
        total_losses.append(total_loss)
        model_1_losses.append(model_1_loss)
        model_2_losses.append(model_2_loss)
        print('Epoch: {:04d}'.format(epoch), 'model_2_loss: {:.4f}'.format(model_2_loss), 'model_1_loss: {:.4f}'.format(model_1_loss),
              'total_loss: {:.4f}'.format(total_loss), 'time: {:.4f}s'.format(time.time() - t))
        torch.save(encoder_2.state_dict(), '{}.k.{}.pth'.format(epoch, args.device))
        torch.save(encoder_1.state_dict(), '{}.q.{}.pth'.format(epoch, args.device))
        if total_losses[-1] < min_loss:
            min_loss = total_losses[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
        
        
        if patience_cnt == args.patience:
            break
        
        
        files_k = glob.glob('*.k.{}.pth'.format(args.device))
        for f in files_k:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)
                print("remove")
        files_q = glob.glob('*.q.{}.pth'.format(args.device))
        for f in files_q:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files_k = glob.glob('*.k.{}.pth'.format(args.device))
    for f in files_k:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    files_q = glob.glob('*.q.{}.pth'.format(args.device))
    for f in files_q:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.4f}'.format(time.time() - t))

    return best_epoch,  model_1_losses, model_2_losses, total_losses

def compute_test(trained_model, dataset):
    """
    Parameters
    ----------
        trained_model : 
        dataset : 

    Outputs
    ------
        acc :
    """

    print('Testing with MLP...')
    trained_model.eval()
    correct = 0

    # split data
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # init
    bad_counter = 0
    best = 10000
    loss_values = []
    best_epoch = 0
    patience = 20
    
    # loss model optimizer
    xent = nn.CrossEntropyLoss()
    log = LogReg(args.nhid, dataset.num_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.001)
    best_acc = torch.zeros(1)

    # cuda
    if torch.cuda.is_available():
        log.to(args.device)
        best_acc = best_acc.to(args.device)

    # train
    for epoch in range(10000):
        log.train()
        opt.zero_grad()
        val_loss = 0.0
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = trained_model(data)
            logits = log(out.detach().clone())
            loss= xent(logits, data.y)
            loss.backward()
            opt.step()
        # val
        for i, data in enumerate(val_loader):
            data = data.to(args.device)
            out = trained_model(data)
            logits = log(out.detach().clone())
            val_loss += xent(logits, data.y).item()
        loss_values.append(val_loss)    
        # save
        torch.save(log.state_dict(), '{}.mlp.{}.pkl'.format(epoch,args.device))   
        # early stop
        if loss_values[-1] < best:
           best = loss_values[-1]
           best_epoch = epoch
           bad_counter = 0
        else:
           bad_counter += 1     
        if bad_counter == patience:
            break
        
        # remove worse pkl
        files = glob.glob('*.mlp.{}.pkl'.format(args.device))
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
               os.remove(file)    
    # remove worse pkl
    files = glob.glob('*.mlp.{}.pkl'.format(args.device))
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    
    print("Optimization Finished!")         
    print('Loading {}th epoch'.format(best_epoch))
    log.load_state_dict(torch.load('{}.mlp.{}.pkl'.format(best_epoch,args.device)))
    # remove all
    files = glob.glob('*.mlp.{}.pkl'.format(args.device))
    for file in files:
            os.remove(file)
    # test
    for i, data in enumerate(test_loader):
            data = data.to(args.device)
            out = trained_model(data)
            logits = log(out.detach().clone())
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(data.y).sum().item()
    acc = correct/num_test
    return acc

def compute_test_svm(trained_model, dataset, search):
    """
    Parameters
    ----------
        trained_model : 
        dataset : 
        search: bool if search C for SVM

    Outputs
    ------
        result :
    """
    # get learned embeddings/labels/split
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
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    # if search C for SVM
    if search:
            params = {"C": [1, 10, 100, 1000, 10000]}
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
            classifier = SVC(C=1)
            classifier.fit(x_train, y_train)
            accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
            return {"Micro-F1": np.mean(accuracies), "std": np.std(accuracies)}
    
def compute_ft_test_svm(finetune_model, dataset):
    finetune_model.eval()
    num_training = int(len(dataset) * 0.9)
    num_val = int(len(dataset) * 0.0)
    num_test = len(dataset) - (num_training + num_val)
  #  data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    accuracies = []
    train_embeddings_list = []
    train_labels_list = [] 
    for i, data in enumerate(train_loader):
            data = data.to(args.device)
            embeddings = finetune_model.embed(data).cpu().numpy()
            labels = data.y.cpu().numpy()
            train_embeddings_list.append(embeddings)
            train_labels_list.append(labels)
    train_embeddings = np.concatenate(tuple(train_embeddings_list), axis=0)
    train_labels = np.concatenate(tuple(train_labels_list), axis=0)
    test_embeddings_list = []
    test_labels_list = [] 
    for i, data in enumerate(test_loader):
            data = data.to(args.device)
            embeddings = finetune_model.embed(data).cpu().numpy()
            labels = data.y.cpu().numpy()
            test_embeddings_list.append(embeddings)
            test_labels_list.append(labels)
    test_embeddings = np.concatenate(tuple(test_embeddings_list), axis=0)
    test_labels = np.concatenate(tuple(test_labels_list), axis=0)
    print(test_embeddings.shape)
    print(train_embeddings.shape)
    classifier = SVC(C=1000)
    classifier.fit(train_embeddings, train_labels)
    acc = accuracy_score(test_labels, classifier.predict(test_embeddings))
    return acc

def finetune_model(pretrained_model, dataset, ft_epoch):
    """
    Parameters
    ----------
        pretrained_model : model with pretrained parameters
        dataset : TUDataset(...)
        ft_epoch: epochs for finetuning

    Outputs
    ------
        finetuned_model : model with finetuned parameters
    """

    correct = 0

     # split data
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # init
    bad_counter = 0
    best = 10000
    loss_values = []
    best_epoch = 0
    patience = 1000
    finetune_epoch = ft_epoch 
    t = time.time()
    best_acc = torch.zeros(1)

    # loss, model, optimizer
    xent = nn.CrossEntropyLoss()
    model = model_ft(args.nhid, dataset.num_classes, pretrained_model)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    
    # cuda
    if torch.cuda.is_available():
        print('Using CUDA')
        model.to(args.device)
        best_acc = best_acc.to(args.device)
    
    # finetune
    for epoch in range(finetune_epoch):
        model.train()
        opt.zero_grad()
        val_loss = 0.0
        # train
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            logits = model(data)
            loss= xent(logits, data.y)
            loss.backward()
            opt.step()
        # val
        for i, data in enumerate(val_loader):
            data = data.to(args.device)
            logits = model(data)
            val_loss += xent(logits, data.y).item()
        loss_values.append(val_loss)    
        # save
        torch.save(model.state_dict(), '{}.ft.{}.pkl'.format(epoch,args.device))   
        # early stop
        if loss_values[-1] < best:
           best = loss_values[-1]
           best_epoch = epoch
           bad_counter = 0
        else:
           bad_counter += 1
        
        if bad_counter == patience:
            break
        # remove pkls worse than best
        files = glob.glob('*.ft.{}.pkl'.format(args.device))
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
               os.remove(file)    
        # print
        print('Epoch: {:04d}'.format(epoch), 'val_loss: {:.4f}'.format(val_loss), 'time: {:.4f}s'.format(time.time() - t)) 
    
    # remove pkls worse than best
    files = glob.glob('*.ft.{}.pkl'.format(args.device))
    best_epoch =finetune_epoch - 1 
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    
    # load best pkl
    print("Optimization Finished!")         
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.ft.{}.pkl'.format(best_epoch,args.device)))
    
    # remove all
    files = glob.glob('*.ft.{}.pkl'.format(args.device))
    for file in files:
        os.remove(file)
    # for i, data in enumerate(test_loader):
    #         data = data.to(args.device)
    #         logits = model(data)
    #         preds = torch.argmax(logits, dim=1)
    #         correct += preds.eq(data.y).sum().item()
    # acc = correct/num_test
    return model.pretrain_model

def infer(trained_model, dataset):
    """
    Parameters
    ----------
        trained_model : 
        dataset : 

    Outputs
    ------
        embeddings :
    """
    # get learned embeddings/labels/split
    trained_model.eval()
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
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

    return embeddings


## test pretrain models
def main_test_pretrained_models(best_epoch=751):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    encoder_1 = GIN(args).to(args.device)       
    encoder_2 = HGP_SL(args).to(args.device)
                
    print('Loading encoder_1 {}th epoch'.format(best_epoch))
    encoder_1.load_state_dict(torch.load('{}.q.{}.pth'.format(best_epoch, args.device)))
    print('Loading encoder_2 {}th epoch'.format(best_epoch))
    encoder_2.load_state_dict(torch.load('{}.k.{}.pth'.format(best_epoch, args.device)))
    encoder_1_acc = compute_test(encoder_1,dataset)
    print('encoder_1 accuracy: {:.4f}'.format(encoder_1_acc))
    encoder_2_acc = compute_test(encoder_2,dataset)
    print('encoder_2 accuracy: {:.4f}'.format(encoder_2_acc))

# Test with svm
def main_test_svm(best_epoch=315):
    encoder_1 = GIN(args).to(args.device)
    encoder_2 = GCN(args).to(args.device)
    print('Loading encoder_1 {}th epoch'.format(best_epoch))
    encoder_1.load_state_dict(torch.load('{}.q.{}.pth'.format(best_epoch, args.device)))
    encoder_1_acc = compute_test_svm(encoder_1, dataset, True)
    print('Loading encoder_2 {}th epoch'.format(best_epoch))
    encoder_2.load_state_dict(torch.load('{}.k.{}.pth'.format(best_epoch, args.device)))
    encoder_2_acc = compute_test_svm(encoder_2, dataset, True)
    print(encoder_1_acc)
    print(encoder_2_acc)
       

## Finetune model with svm
def main_finetune_with_svm(best_epoch=73):        
    model_acc_collec = []
    pretrained_model = GCN(args).to(args.device)
    for i in range(5):    
        pretrained_model.reset_parameters()

        print('Loading encoder_2 {}th epoch'.format(best_epoch))
        pretrained_model.load_state_dict(torch.load('{}.k.{}.pth'.format(best_epoch, args.device)))
        ft_model = finetune_model(pretrained_model, dataset, 120)
        #finetune_model = GIN_ft(args.nhid, dataset.num_classes, ).to(args.device)
        #finetune_model.load_state_dict(torch.load('{}.ft.{}.pth'.format(best_epoch, args.device))) 
        acc = compute_test_svm(ft_model, dataset, True)
        print(acc)
        model_acc_collec.append(acc)
    model_acc_mean = torch.Tensor(model_acc_collec).mean()
    print('model accuracy mean: {:.4f}'.format(model_acc_mean))    
    model_acc_std = torch.Tensor(model_acc_collec).std()
    print('model accuracy std: {:.4f}'.format(model_acc_std))

## Finetune with MLP
def main_finetune_with_mlp(best_epoch = 442):        
    pretrained_model = GIN(args).to(args.device)
            
    print('Loading encoder_1 {}th epoch'.format(best_epoch))
    pretrained_model.load_state_dict(torch.load('{}.q.{}.pth'.format(best_epoch, args.device)))
    acc = finetune_model(pretrained_model,dataset)
    print(acc)

# vis
def vis(best_epoch = 483):
    encoder_1 = GIN(args).to(args.device)
    encoder_2 = GCN(args).to(args.device)

    print(torch.load('./checkpoints/{}.q.{}.pth'.format(best_epoch, args.device)))
    encoder_1.load_state_dict(torch.load('./checkpoints/{}.q.{}.pth'.format(best_epoch, args.device)))
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
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 20, labels[:])
    plt.savefig("./pics/{}_gin.jpg".format(args.dataset),dpi=50)


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
    return encoder_1, encoder_2


if __name__ == '__main__':
        encoder_2_acc_collec , encoder_1_acc_collec = [], []
        corr_pearson_collec, corr_spear_collec = [], []
        stop_loss_collec = []

        for repeat_idx in range(args.repeat_times):
            ###print('Repeation:{}'.format(repeat_idx))
            # prepare data
            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            encoder_1, encoder_2 = init_encoders(args)
            
            optimizer_1 = torch.optim.Adam(encoder_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_2 = torch.optim.Adam(encoder_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            
            # Model training
            stdv = 1.0 / math.sqrt(args.num_features / 3)
            model_2_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            model_1_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            #model_2_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            #model_1_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            best_epoch,  model_1_losses, model_2_losses, total_losses = train_two_models(encoder_1, encoder_2, optimizer_1, optimizer_2, model_1_mb, model_2_mb, data_loader)
            
            ###print('Loading trained encoder 1 from {}'.format(save_path_1))
            ###print('Loading trained encoder 2 from {}'.format(save_path_2))

            encoder_1.load_state_dict(torch.load(save_path_1))
            encoder_2.load_state_dict(torch.load(save_path_2))

            encoder_1_acc = compute_test_svm(encoder_1,dataset,search=True)
            encoder_2_acc = compute_test_svm(encoder_2,dataset,search=True)

            encoder_1_acc_collec.append(encoder_1_acc)
            encoder_2_acc_collec.append(encoder_2_acc)

            ###print('encoder_1 acc: {:.4f}'.format(encoder_1_acc))
            ###print('encoder_2 acc: {:.4f}'.format(encoder_2_acc))

            # assembly metric
            corr_pearson, corr_spear = \
                assembly_metrics.compute_double_rdm(encoder_1,encoder_2,dataset,args)

            corr_pearson_collec.append(corr_pearson)
            corr_spear_collec.append(corr_spear)

            stop_loss_collec.append(total_losses[-1]/(2*len(dataset)))
        
        print('---------------------------------------------------------------------')
        print('Dataset:{} | assembly:{} | encoder_1:{} | encoder_2:{}'.format(args.dataset,assembly,args.encoder_1,args.encoder_2))
        print('Average encoder_1 acc: {:.4f}±{:.4f}'.format(np.mean(encoder_1_acc_collec), np.std(encoder_1_acc_collec)))
        print('Average encoder_2 acc: {:.4f}±{:.4f}'.format(np.mean(encoder_2_acc_collec), np.std(encoder_2_acc_collec)))
        print('---------Assembly Metrics---------')
        print('pearson rdm: {:.4f}±{:.4f}'.format(np.mean(corr_pearson_collec), np.std(corr_pearson_collec)))
        print('spear rdm: {:.4f}±{:.4f}'.format(np.mean(corr_spear_collec), np.std(corr_spear_collec)))
        print('stop loss: {:.4f}±{:.4f}'.format(np.mean(stop_loss_collec), np.std(stop_loss_collec)))

     

#GCN+GIN
if 0:
        encoder_2_acc_collec , encoder_1_acc_collec = [], []
        for i in range(1):
            # prepare data
            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            encoder_1 = GIN(args).to(args.device)
            
            encoder_2 = GCN(args).to(args.device)
            
            optimizer_2 = torch.optim.Adam(encoder_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_1 = torch.optim.Adam(encoder_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            #optimizer = torch.optim.Adam(list(encoder_2.parameters()) + list(encoder_1.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            # Model training
            
            stdv = 1.0 / math.sqrt(args.num_features / 3)
            model_2_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            model_1_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            #model_2_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            #model_1_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            best_epoch,  model_1_losses, model_2_losses, total_losses = train_two_models(encoder_1, encoder_2, optimizer_1, optimizer_2,   model_1_mb, model_2_mb,data_loader)
            
            print('Loading encoder_1 {}th epoch'.format(best_epoch))
            encoder_1.load_state_dict(torch.load('{}.q.{}.pth'.format(best_epoch, args.device)))
            print('Loading encoder_2 {}th epoch'.format(best_epoch))
            encoder_2.load_state_dict(torch.load('{}.k.{}.pth'.format(best_epoch, args.device)))
            encoder_1_acc = compute_test(encoder_1,dataset)
            print('encoder_1 accuracy: {:.4f}'.format(encoder_1_acc))
            encoder_1_acc_collec.append(encoder_1_acc)
            encoder_2_acc = compute_test(encoder_2,dataset)
            print('encoder_2 accuracy: {:.4f}'.format(encoder_2_acc))
            encoder_2_acc_collec.append(encoder_2_acc)
        encoder_1_acc_mean = torch.Tensor(encoder_1_acc_collec).mean()
        print('encoder_1 accuracy mean: {:.4f}'.format(encoder_1_acc_mean))
        encoder_1_acc_std = torch.Tensor(encoder_1_acc_collec).std()
        print('encoder_1 accuracy std: {:.4f}'.format(encoder_1_acc_std))
        encoder_2_acc_mean = torch.Tensor(encoder_2_acc_collec).mean()
        print('encoder_2 accuracy mean: {:.4f}'.format(encoder_2_acc_mean))
        encoder_2_acc_std = torch.Tensor(encoder_2_acc_collec).std()
        print('encoder_2 accuracy std: {:.4f}'.format(encoder_2_acc_std))

#DGCNN(sortpool)+GIN
if 0:
        encoder_2_acc_collec , encoder_1_acc_collec = [], []
        for i in range(10):
            # prepare data
            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            encoder_1 = GIN(args).to(args.device)
            
            encoder_2 = SortPool(args).to(args.device)
            
            optimizer_2 = torch.optim.Adam(encoder_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_1 = torch.optim.Adam(encoder_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            #optimizer = torch.optim.Adam(list(encoder_2.parameters()) + list(encoder_1.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            # Model training
            
            stdv = 1.0 / math.sqrt(args.num_features / 3)
            model_2_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            model_1_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            #model_2_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            #model_1_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            best_epoch,  model_1_losses, model_2_losses, total_losses = train_two_models(encoder_1, encoder_2,optimizer_1, optimizer_2,  model_1_mb, model_2_mb, data_loader)
            
            print('Loading encoder_1 {}th epoch'.format(best_epoch))
            encoder_1.load_state_dict(torch.load('{}.q.{}.pth'.format(best_epoch, args.device)))
            print('Loading encoder_2 {}th epoch'.format(best_epoch))
            encoder_2.load_state_dict(torch.load('{}.k.{}.pth'.format(best_epoch, args.device)))
            encoder_1_acc = compute_test(encoder_1,dataset)
            print('encoder_1 accuracy: {:.4f}'.format(encoder_1_acc))
            encoder_1_acc_collec.append(encoder_1_acc)
            encoder_2_acc = compute_test(encoder_2,dataset)
            print('encoder_2 accuracy: {:.4f}'.format(encoder_2_acc))
            encoder_2_acc_collec.append(encoder_2_acc)
        encoder_1_acc_mean = torch.Tensor(encoder_1_acc_collec).mean()
        print('encoder_1 accuracy mean: {:.4f}'.format(encoder_1_acc_mean))
        encoder_1_acc_std = torch.Tensor(encoder_1_acc_collec).std()
        print('encoder_1 accuracy std: {:.4f}'.format(encoder_1_acc_std))
        encoder_2_acc_mean = torch.Tensor(encoder_2_acc_collec).mean()
        print('encoder_2 accuracy mean: {:.4f}'.format(encoder_2_acc_mean))
        encoder_2_acc_std = torch.Tensor(encoder_2_acc_collec).std()
        print('encoder_2 accuracy std: {:.4f}'.format(encoder_2_acc_std))

#GAT+GIN
if 0:
        encoder_2_acc_collec , encoder_1_acc_collec = [], []
        for i in range(10):
            # prepare data
            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            encoder_1 = GIN(args).to(args.device)
            
            encoder_2 = GAT(args).to(args.device)
            
            optimizer_1 = torch.optim.Adam(encoder_1.parameters(), lr=args.lr_q, weight_decay=args.weight_decay)
            optimizer_2 = torch.optim.Adam(encoder_2.parameters(), lr=args.lr_k, weight_decay=args.weight_decay)
            
            
            #optimizer = torch.optim.Adam(list(encoder_2.parameters()) + list(encoder_1.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            # Model training
            
            stdv = 1.0 / math.sqrt(args.num_features / 3)
            model_2_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            model_1_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            #model_2_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            #model_1_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            best_epoch,  model_1_losses, model_2_losses, total_losses = train_two_models(encoder_1, encoder_2, optimizer_1, optimizer_2, model_1_mb, model_2_mb, data_loader)
            
            print('Loading encoder_1 {}th epoch'.format(best_epoch))
            encoder_1.load_state_dict(torch.load('{}.q.{}.pth'.format(best_epoch, args.device)))
            print('Loading encoder_2 {}th epoch'.format(best_epoch))
            encoder_2.load_state_dict(torch.load('{}.k.{}.pth'.format(best_epoch, args.device)))
            encoder_1_acc = compute_test(encoder_1,dataset)
            print('encoder_1 accuracy: {:.4f}'.format(encoder_1_acc))
            encoder_1_acc_collec.append(encoder_1_acc)
            encoder_2_acc = compute_test(encoder_2,dataset)
            print('encoder_2 accuracy: {:.4f}'.format(encoder_2_acc))
            encoder_2_acc_collec.append(encoder_2_acc)
        encoder_1_acc_mean = torch.Tensor(encoder_1_acc_collec).mean()
        print('encoder_1 accuracy mean: {:.4f}'.format(encoder_1_acc_mean))
        encoder_1_acc_std = torch.Tensor(encoder_1_acc_collec).std()
        print('encoder_1 accuracy std: {:.4f}'.format(encoder_1_acc_std))
        encoder_2_acc_mean = torch.Tensor(encoder_2_acc_collec).mean()
        print('encoder_2 accuracy mean: {:.4f}'.format(encoder_2_acc_mean))
        encoder_2_acc_std = torch.Tensor(encoder_2_acc_collec).std()
        print('encoder_2 accuracy std: {:.4f}'.format(encoder_2_acc_std))


#GCN+GIN dual temperature + SVM
if 0:
        encoder_2_acc_collec , encoder_1_acc_collec = [], []
        for i in range(1):
            # prepare data
            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            encoder_1 = GIN(args).to(args.device)
            
            encoder_2 = SortPool(args).to(args.device)
            
            optimizer_2 = torch.optim.Adam(encoder_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_1 = torch.optim.Adam(encoder_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            #optimizer = torch.optim.Adam(list(encoder_2.parameters()) + list(encoder_1.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            # Model training
            
            stdv = 1.0 / math.sqrt(args.num_features / 3)
            model_2_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            model_1_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            #model_2_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            #model_1_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            best_epoch,  model_1_losses, model_2_losses, total_losses = train_two_models_dual_temperature(encoder_1, encoder_2, optimizer_2, optimizer_1, data_loader)
            
            print('Loading encoder_1 {}th epoch'.format(best_epoch))
            encoder_1.load_state_dict(torch.load('{}.q.{}.pth'.format(best_epoch, args.device)))
            print('Loading encoder_2 {}th epoch'.format(best_epoch))
            encoder_2.load_state_dict(torch.load('{}.k.{}.pth'.format(best_epoch, args.device)))
            encoder_1_acc = compute_test_svm(encoder_1,dataset, True)
            print(encoder_1_acc)
           # encoder_1_acc_collec.append(encoder_1_acc)
            encoder_2_acc = compute_test_svm(encoder_2,dataset, True)
            print(encoder_2_acc)
           # encoder_2_acc_collec.append(encoder_2_acc)
        # encoder_1_acc_mean = torch.Tensor(encoder_1_acc_collec).mean()
        # print('encoder_1 accuracy mean: {:.4f}'.format(encoder_1_acc_mean))
        # encoder_1_acc_std = torch.Tensor(encoder_1_acc_collec).std()
        # print('encoder_1 accuracy std: {:.4f}'.format(encoder_1_acc_std))
        # encoder_2_acc_mean = torch.Tensor(encoder_2_acc_collec).mean()
        # print('encoder_2 accuracy mean: {:.4f}'.format(encoder_2_acc_mean))
        # encoder_2_acc_std = torch.Tensor(encoder_2_acc_collec).std()
        # print('encoder_2 accuracy std: {:.4f}'.format(encoder_2_acc_std))


#GCN+GIN one temperature + SVM
if 0:
        encoder_2_acc_collec , encoder_1_acc_collec = [], []
        for i in range(5):
            # prepare data
            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            encoder_1 = GIN(args).to(args.device)
            
            encoder_2 = GCN(args).to(args.device)
            
            optimizer_2 = torch.optim.Adam(encoder_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_1 = torch.optim.Adam(encoder_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # optimizer = torch.optim.Adam(list(encoder_2.parameters()) + list(encoder_1.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            # Model training
            
            stdv = 1.0 / math.sqrt(args.num_features / 3)
            model_2_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            model_1_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            #model_2_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            #model_1_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            best_epoch,  model_1_losses, model_2_losses, total_losses = train_two_models(encoder_1, encoder_2, optimizer_1, optimizer_2, model_1_mb, model_2_mb, data_loader)
            
            print('Loading encoder_1 {}th epoch'.format(best_epoch))
            encoder_1.load_state_dict(torch.load('{}.q.{}.pth'.format(best_epoch, args.device)))
            print('Loading encoder_2 {}th epoch'.format(best_epoch))
            encoder_2.load_state_dict(torch.load('{}.k.{}.pth'.format(best_epoch, args.device)))
            encoder_1_acc = compute_test_svm(encoder_1,dataset, True)   
            print(encoder_1_acc)
            encoder_1_acc_collec.append(encoder_1_acc)
            encoder_2_acc = compute_test_svm(encoder_2,dataset, True)
            print(encoder_2_acc)
            encoder_2_acc_collec.append(encoder_2_acc)
        print('dataset: {}'.format(args.dataset))
        print(args)
        encoder_1_acc_mean = torch.Tensor(encoder_1_acc_collec).mean()
        print('encoder_1 accuracy mean: {:.4f}'.format(encoder_1_acc_mean))
        encoder_1_acc_std = torch.Tensor(encoder_1_acc_collec).std()
        print('encoder_1 accuracy std: {:.4f}'.format(encoder_1_acc_std))
        encoder_2_acc_mean = torch.Tensor(encoder_2_acc_collec).mean()
        print('encoder_2 accuracy mean: {:.4f}'.format(encoder_2_acc_mean))
        encoder_2_acc_std = torch.Tensor(encoder_2_acc_collec).std()
        print('encoder_2 accuracy std: {:.4f}'.format(encoder_2_acc_std))

#GCN+GIN one temperature + SVM + Memory
if 0:
        encoder_2_acc_collec , encoder_1_acc_collec = [], []
        for i in range(3):
            # prepare data
            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            encoder_1 = GIN(args).to(args.device)
            
            encoder_2 = GCN(args).to(args.device)
            
            optimizer_2 = torch.optim.Adam(encoder_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_1 = torch.optim.Adam(encoder_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # optimizer = torch.optim.Adam(list(encoder_2.parameters()) + list(encoder_1.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            # Model training
            
            stdv = 1.0 / math.sqrt(args.num_features / 3)
            model_2_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            model_1_mb = torch.rand(len(dataset), args.nhid).mul_(2 * stdv).add_(-stdv).to(args.device)
            # model_2_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            # model_1_mb = torch.zeros(len(dataset), args.nhid).to(args.device)
            best_epoch, model_1_losses, model_2_losses, total_losses = train_two_models_with_memery(encoder_1, encoder_2, optimizer_2, optimizer_1, model_2_mb, model_1_mb, data_loader)
            
            print('Loading encoder_1 {}th epoch'.format(best_epoch))
            encoder_1.load_state_dict(torch.load('{}.q.{}.pth'.format(best_epoch, args.device)))
            print('Loading encoder_2 {}th epoch'.format(best_epoch))
            encoder_2.load_state_dict(torch.load('{}.k.{}.pth'.format(best_epoch, args.device)))
            encoder_1_acc = compute_test_svm(encoder_1,dataset, True)   
            print(encoder_1_acc)
            encoder_1_acc_collec.append(encoder_1_acc)
            encoder_2_acc = compute_test_svm(encoder_2,dataset, True)
            print(encoder_2_acc)
            encoder_2_acc_collec.append(encoder_2_acc)
        print('dataset: {}'.format(args.dataset))
        print(args)
        encoder_1_acc_mean = torch.Tensor(encoder_1_acc_collec).mean()
        print('encoder_1 accuracy mean: {:.4f}'.format(encoder_1_acc_mean))
        encoder_1_acc_std = torch.Tensor(encoder_1_acc_collec).std()
        print('encoder_1 accuracy std: {:.4f}'.format(encoder_1_acc_std))
        encoder_2_acc_mean = torch.Tensor(encoder_2_acc_collec).mean()
        print('encoder_2 accuracy mean: {:.4f}'.format(encoder_2_acc_mean))
        encoder_2_acc_std = torch.Tensor(encoder_2_acc_collec).std()
        print('encoder_2 accuracy std: {:.4f}'.format(encoder_2_acc_std))


