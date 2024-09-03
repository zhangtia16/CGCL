# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

@torch.no_grad()
def momentum_update(encoder_q, encoder_k, m=0.999):
    """
    encoder_k = m * encoder_k + (1 - m) encoder_q
    """        
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

    return encoder_k


def dequeue_and_enqueue(hidden_batch_feats, selected_batch_idx, memory_bank):
    '''
    update memory bank by batch window slide; hidden_batch_feats must be normalized
    '''
#    assert(hidden_batch_feats.size()[1] == memory_bank.size()[1])
    memory_bank[selected_batch_idx] = nn.functional.normalize(hidden_batch_feats,dim=1)
    #memory_bank[selected_batch_idx] = hidden_batch_feats
    return memory_bank

def update_whole_memory_bank(args, model, memory_bank, train_loader):
    '''
    update memory bank by batch window slide; hidden_batch_feats must be normalized
    '''
#    assert(hidden_batch_feats.size()[1] == memory_bank.size()[1])
    with torch.no_grad():
        start_point = 0
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            learned_representations = nn.functional.normalize(model.embed(data),dim=1)
            query_index = torch.arange(start_point, start_point+len(data.y))
            memory_bank = dequeue_and_enqueue(learned_representations, query_index, memory_bank)
            start_point += len(data.y)
    return memory_bank


