from __future__ import print_function
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math
import numpy as np
import pandas as pd
from torch.optim import Optimizer
import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp

sys.path.insert(0,'./../utils/')
from logger import *
from eval import *
from misc import *

from cifar10_normal_train import *
from cifar10_util import *
from adam import Adam
from sgd import SGD


# ## Get cifar10 data and split it in IID fashion

# In[2]:


import torchvision.transforms as transforms
import torchvision.datasets as datasets
data_loc='/mnt/nfs/work1/amir/vshejwalkar/cifar10_data/'

# ## Code for Multi-krum aggregation algorithm

# In[4]:


def multi_krum(all_updates, n_attackers, multi_k=False):

    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    # print(len(remaining_updates))

    aggregate = torch.mean(candidates, dim=0)

    return aggregate, np.array(candidate_indices)


# ## Code for Bulyan aggregation algorithm

# In[5]:


def bulyan(all_updates, n_attackers):
    nusers = all_updates.shape[0]
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(bulyan_cluster) < (nusers - 2 * n_attackers):
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        # print(distances)

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        if not len(indices):
            break
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    # print('dim of bulyan cluster ', bulyan_cluster.shape)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0), np.array(candidate_indices)

# ## Code for fang attack on Bulyan
# ### Note that Fang attacks on Multi-krum and Bulyan are the same

# In[6]:


def compute_lambda_fang(all_updates, model_re, n_attackers):

    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1)
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0])
    max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])

    return (term_1 + max_wre_dist)


def get_malicious_updates_fang(all_updates, model_re, deviation, n_attackers):

    lamda = compute_lambda_fang(all_updates, model_re, n_attackers)
    threshold = 1e-5

    mal_updates = []
    while lamda > threshold:
        mal_update = (- lamda * deviation)

        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=False)
        
        if krum_candidate < n_attackers:
            return mal_update
        
        lamda *= 0.5

    if not len(mal_updates):
        print(lamda, threshold)
        mal_update = (model_re - lamda * deviation)

    return mal_update


def tr_mean(all_updates, n_attackers):
    sorted_updates = torch.sort(all_updates, 0)[0]
    out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
    return out

def get_malicious_updates_fang_trmean(all_updates, deviation, n_attackers, epoch_num, compression='none', q_level=2, norm='inf'):
    b = 2
    max_vector = torch.max(all_updates, 0)[0]
    min_vector = torch.min(all_updates, 0)[0]

    max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
    min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
    min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    quant_mal_vec = []
    if compression != 'none':
        if epoch_num == 0: print('compressing malicious update')
        for i in range(mal_vec.shape[0]):
            mal_ = mal_vec[i]
            mal_quant = qsgd(mal_, s=q_level, norm=norm)
            quant_mal_vec = mal_quant[None, :] if not len(quant_mal_vec) else torch.cat((quant_mal_vec, mal_quant[None, :]), 0)
    else:
        quant_mal_vec = mal_vec

    mal_updates = torch.cat((quant_mal_vec, all_updates), 0)

    return mal_updates


# ## Execute Fang attack on Bulyan

# In[7]:


def fang_bulyan_unknown_benign(nepochs=1200, fed_lr=0.5):
    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    fed_lr=0.5
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='fang'
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0
        
        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(user_grads[:n_attacker], agg_grads, deviation, n_attacker_)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_median(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)
            
            mal_updates = torch.stack([mal_update] * n_attacker)
            malicious_grads = torch.cat((mal_updates, user_grads), 0)
            
            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()
                
            agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc), flush=True)

            epoch_num+=1

def lie_bulyan_unknown_benign(nepochs=1200, fed_lr=0.5):
    def lie_attack(all_updates, z):
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='lie'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                
            mal_updates = torch.stack([mal_update] * n_attacker)
            malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()
                
            agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def agr_tailored_bulyan_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_mkrum(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([3.0]).cuda()

        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)
            if np.sum(krum_candidate < n_attackers) == n_attackers:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        return mal_update

    batch_size=250
    resume=0
    nepochs=1200
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    fed_lr=0.5
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='our-agr'
    dev_type ='std'
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation


    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(user_grads[:n_attacker], agg_grads, deviation, n_attacker_)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_mkrum(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)
                
            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()
            
            agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_max_bulyan_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='min-max'
    dev_type ='std'
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(user_grads[:n_attacker], agg_grads, deviation, n_attacker_)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_mkrum(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)

            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()
            
            agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_sum_bulyan_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_score(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        
        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='min-sum'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(user_grads[:n_attacker], agg_grads, deviation, n_attacker_)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_mkrum(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)

            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()
            
            agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def fang_bulyan_known_benign(nepochs=1200, fed_lr=0.5):
    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='fang'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_file='%s_checkpoint_%s_%d.pth.tar'%(arch, at_type, n_attacker)
        fed_best_file='%s_best_%s_%d.pth.tar'%(arch, at_type, n_attacker)
        
        if resume:
            fed_checkpoint = chkpt+'/'+fed_file
            assert os.path.isfile(fed_checkpoint), 'Error: no user checkpoint at %s'%(fed_checkpoint)
            checkpoint = torch.load(fed_checkpoint, map_location='cuda:%d'%torch.cuda.current_device())
            fed_model.load_state_dict(checkpoint['state_dict'])
            optimizer_fed.load_state_dict(checkpoint['optimizer'])
            resume = 0
            best_global_acc=checkpoint['best_acc']
            best_global_te_acc=checkpoint['best_te_acc']
            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            epoch_num += checkpoint['epoch']
            print('resuming from epoch %d | val acc %.4f | best acc %.3f | best te acc %.3f'%(epoch_num, val_acc, best_global_acc, best_global_te_acc))

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'paf':
                    malicious_grads=get_malicious_predictions_poison_all_far_sign(malicious_grads,nusers,n_attacker)
                elif at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_krum(malicious_grads, agg_grads, n_attacker, compression, q_level, norm)

            if not epoch_num:
                print(malicious_grads.shape)
                
            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':

                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0:
                    print('multi krum is ', multi_k)

                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)


            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%1==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            epoch_num+=1

def lie_bulyan_known_benign(nepochs=1200, fed_lr=0.5):
    def lie_attack(all_updates, z):
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std

    batch_size=250
    resume=0
    nepochs=1200
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    fed_lr=0.5
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='LIE'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                    malicious_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_krum(malicious_grads, agg_grads, n_attacker, compression=compression, q_level=q_level, norm=norm)

            if not epoch_num : 
                print(malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='trmean':
                agg_grads=tr_mean(malicious_grads, n_attacker)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def agr_tailored_bulyan_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_mkrum(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([3.0]).cuda()

        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)
            if np.sum(krum_candidate < n_attackers) == n_attackers:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        return mal_updates

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='our-agr'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation


    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_mkrum(malicious_grads, agg_grads, n_attacker, dev_type=dev_type)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':

                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%1==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_max_bulyan_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='min-max'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_score(malicious_grads, agg_grads, n_attacker, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%1==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_sum_bulyan_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_score(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        
        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='bulyan'
    multi_k = False
    candidates = []

    at_type='min-sum'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_score(malicious_grads, agg_grads, n_attacker, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%1==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def fang_median_unknown_benign(nepochs=1200, fed_lr=0.5):
    def fang_attack_trmean_partial(all_updates, n_attackers):

        model_re = torch.mean(all_updates, 0)
        model_std = torch.std(all_updates, 0)
        deviation = torch.sign(model_re)
        
        max_vector_low = model_re + 3 * model_std 
        max_vector_hig = model_re + 4 * model_std
        min_vector_low = model_re - 4 * model_std
        min_vector_hig = model_re - 3 * model_std

        max_range = torch.cat((max_vector_low[:,None], max_vector_hig[:,None]), dim=1)
        min_range = torch.cat((min_vector_low[:,None], min_vector_hig[:,None]), dim=1)

        rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

        max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
        min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

        mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
            [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

        return mal_vec

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'
    multi_k = False
    candidates = []

    at_type='fang'
    z_values=[0.0]
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grad = get_malicious_updates_lie(user_grads[:n_attacker], n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    mal_updates = fang_attack_trmean_partial(user_grads[:n_attacker], n_attacker)

            malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]))
            
            if not epoch_num : 
                print(malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='trmean':
                agg_grads=tr_mean(malicious_grads, n_attacker)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def lie_median_unknown_benign(nepochs=1200, fed_lr=0.5):
    def lie_attack(all_updates, z):
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'
    multi_k = False
    candidates = []

    at_type='LIE'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                    mal_updates = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    mal_updates = fang_attack_trmean_partial(user_grads[:n_attacker], n_attacker)

            malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]))
            
            if not epoch_num : 
                print(malicious_grads.shape)

            agg_grads=torch.median(malicious_grads,dim=0)[0]

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def agr_tailored_median_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_median(all_updates, model_re, n_attackers, dev_type='unit_vec', threshold=30):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([threshold]).cuda() #compute_lambda_our(all_updates, model_re, n_attackers)

        threshold_diff = 1e-5
        prev_loss = -1
        lamda_fail = lamda
        lamda_succ = 0
        iters = 0 
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads = torch.median(mal_updates, 0)[0]
            
            loss = torch.norm(agg_grads - model_re)
            
            if prev_loss < loss:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
            prev_loss = loss
            
        mal_update = (model_re - lamda_succ * deviation)
        return mal_update
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        return mal_updates

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'
    multi_k = False
    candidates = []

    at_type='our-agr'
    dev_type = 'std'
    threshold=10
    n_attackers=[10]
    partial_attackers = {4:1, 5:1, 8:2, 10:3, 12:4}

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                    mal_updates = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'our-agr':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_median(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                    mal_updates = torch.stack([mal_update]*n_attacker)
                    
            malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]))
            
            if not epoch_num :  print(malicious_grads.shape)

            agg_grads=torch.median(malicious_grads,dim=0)[0]

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def min_max_median_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec', threshold=30):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([threshold]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'
    multi_k = False
    candidates = []

    at_type='min-max'
    dev_type ='std'
    threshold=10
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    mal_updates = fang_attack_trmean_partial(user_grads[:n_attacker], n_attacker)
                elif at_type == 'our-agr':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_median(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                elif at_type == 'min-max':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                elif at_type == 'min-sum':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            agg_grads=torch.median(malicious_grads,dim=0)[0]

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d | e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_sum_median_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_score(all_updates, model_re, n_attackers, dev_type='unit_vec', threshold=30):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        
        lamda = torch.Tensor([threshold]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'

    at_type='min-sum'
    dev_type ='std'
    threshold=10
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                    attacker_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    attacker_grads = fang_attack_trmean_partial(user_grads[:n_attacker], n_attacker)
                elif at_type == 'our-agr':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_median(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                elif at_type == 'min-max':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                elif at_type == 'min-sum':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            agg_grads=torch.median(malicious_grads,dim=0)[0]

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d | e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def fang_median_known_benign(nepochs=1200, fed_lr=0.5):
    def get_malicious_updates_fang_trmean(all_updates, deviation, n_attackers, epoch_num, compression='none', q_level=2, norm='inf'):
        b = 2
        max_vector = torch.max(all_updates, 0)[0]
        min_vector = torch.min(all_updates, 0)[0]

        max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
        min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

        max_[max_ == 1] = b
        max_[max_ == 0] = 1 / b
        min_[min_ == 1] = b
        min_[min_ == 0] = 1 / b

        max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
        min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

        rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

        max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
        min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

        mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
            [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

        quant_mal_vec = []
        if compression != 'none':
            if epoch_num == 0: print('compressing malicious update')
            for i in range(mal_vec.shape[0]):
                mal_ = mal_vec[i]
                mal_quant = qsgd(mal_, s=q_level, norm=norm)
                quant_mal_vec = mal_quant[None, :] if not len(quant_mal_vec) else torch.cat((quant_mal_vec, mal_quant[None, :]), 0)
        else:
            quant_mal_vec = mal_vec

        mal_updates = torch.cat((quant_mal_vec, all_updates), 0)

        return mal_updates

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'
    multi_k = False
    candidates = []

    at_type='fang'
    z_values=[0.0]
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        for z in z_values:
            fed_file='alexnet_checkpoint_%s_%s_%d_%.2f.pth.tar'%(aggregation,at_type,n_attacker,z)
            fed_best_file='alexnet_best_%s_%s_%d_%.2f.pth.tar'%(aggregation,at_type,n_attacker,z)

            if resume:
                fed_checkpoint = chkpt+'/'+fed_file
                assert os.path.isfile(fed_checkpoint), 'Error: no user checkpoint at %s'%(fed_checkpoint)
                checkpoint = torch.load(fed_checkpoint, map_location='cuda:%d'%torch.cuda.current_device())
                fed_model.load_state_dict(checkpoint['state_dict'])
                optimizer_fed.load_state_dict(checkpoint['optimizer'])
                resume = 0
                best_global_acc=checkpoint['best_acc']
                best_global_te_acc=checkpoint['best_te_acc']
                val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
                epoch_num += checkpoint['epoch']
                print('resuming from epoch %d | val acc %.4f | best acc %.3f | best te acc %.3f'%(epoch_num, val_acc, best_global_acc, best_global_te_acc))

            torch.cuda.empty_cache()
            r=np.arange(user_tr_len)

            fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
            optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

            while epoch_num <= nepochs:
                user_grads=[]
                if not epoch_num and epoch_num%nbatches == 0:
                    np.random.shuffle(r)
                    for i in range(nusers):
                        user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                        user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

                for i in range(n_attacker, nusers):

                    inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                    targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                    inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                    outputs = fed_model(inputs)
                    loss = criterion(outputs, targets)
                    fed_model.zero_grad()
                    loss.backward(retain_graph=True)

                    param_grad=[]
                    for param in fed_model.parameters():
                        param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                    user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

                malicious_grads = user_grads

                if epoch_num in schedule:
                    for param_group in optimizer_fed.param_groups:
                        param_group['lr'] *= gamma
                        print('New learnin rate ', param_group['lr'])

                if n_attacker > 0:
                    if at_type == 'paf':
                        malicious_grads=get_malicious_predictions_poison_all_far_sign(malicious_grads,nusers,n_attacker)
                    elif at_type == 'lie':
                        malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                    elif at_type == 'fang':
                        agg_grads = torch.mean(malicious_grads, 0)
                        deviation = torch.sign(agg_grads)
                        malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
                    elif at_type == 'our':
                        agg_grads = torch.mean(malicious_grads, 0)
                        malicious_grads = our_attack_krum(malicious_grads, agg_grads, n_attacker, compression=compression, q_level=q_level, norm=norm)

                if not epoch_num : 
                    print(malicious_grads.shape)
                    
                if aggregation=='median':
                    agg_grads=torch.median(malicious_grads,dim=0)[0]

                elif aggregation=='average':
                    agg_grads=torch.mean(malicious_grads,dim=0)

                elif aggregation=='trmean':
                    agg_grads=tr_mean(malicious_grads, n_attacker)

                elif aggregation=='krum' or aggregation=='mkrum':
                    multi_k = True if aggregation == 'mkrum' else False
                    if epoch_num == 0: print('multi krum is ', multi_k)
                    agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)
                    
                elif aggregation=='bulyan':
                    agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

                del user_grads

                start_idx=0

                optimizer_fed.zero_grad()

                model_grads=[]

                for i, param in enumerate(fed_model.parameters()):
                    param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                    start_idx=start_idx+len(param.data.view(-1))
                    param_=param_.cuda()
                    model_grads.append(param_)

                optimizer_fed.step(model_grads)

                val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
                te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

                is_best = best_global_acc < val_acc

                best_global_acc = max(best_global_acc, val_acc)

                if is_best:
                    best_global_te_acc = te_acc

                if epoch_num%10==0 or epoch_num==nepochs-1:
                    print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

                if val_loss > 10:
                    print('val loss %f too high'%val_loss)
                    break
                
                epoch_num+=1

def lie_median_known_benign(nepochs=1200, fed_lr=0.5):
    def lie_attack(all_updates, z):
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'
    multi_k = False
    candidates = []

    at_type='LIE'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

    #     fed_file='alexnet_checkpoint_%s_%s_%d_%.2f.pth.tar'%(aggregation,at_type,n_attacker,z)
    #     fed_best_file='alexnet_best_%s_%s_%d_%.2f.pth.tar'%(aggregation,at_type,n_attacker,z)

    #     if resume:
    #         fed_checkpoint = chkpt+'/'+fed_file
    #         assert os.path.isfile(fed_checkpoint), 'Error: no user checkpoint at %s'%(fed_checkpoint)
    #         checkpoint = torch.load(fed_checkpoint, map_location='cuda:%d'%torch.cuda.current_device())
    #         fed_model.load_state_dict(checkpoint['state_dict'])
    #         optimizer_fed.load_state_dict(checkpoint['optimizer'])
    #         resume = 0
    #         best_global_acc=checkpoint['best_acc']
    #         best_global_te_acc=checkpoint['best_te_acc']
    #         val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
    #         epoch_num += checkpoint['epoch']
    #         print('resuming from epoch %d | val acc %.4f | best acc %.3f | best te acc %.3f'%(epoch_num, val_acc, best_global_acc, best_global_te_acc))

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                    malicious_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_krum(malicious_grads, agg_grads, n_attacker, compression=compression, q_level=q_level, norm=norm)

            if not epoch_num : 
                print(malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='trmean':
                agg_grads=tr_mean(malicious_grads, n_attacker)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def agr_tailored_median_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_median(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).cuda() #compute_lambda_our(all_updates, model_re, n_attackers)

        threshold_diff = 1e-5
        prev_loss = -1
        lamda_fail = lamda
        lamda_succ = 0
        iters = 0 
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads = torch.median(mal_updates, 0)[0]
            
            loss = torch.norm(agg_grads - model_re)
            
            if prev_loss < loss:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
            prev_loss = loss
            
        mal_update = (model_re - lamda_succ * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        return mal_updates

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'
    multi_k = False
    candidates = []

    at_type='our-agr'
    dev_type = 'std'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                    malicious_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type=dev_type)

            if not epoch_num : 
                print(malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='trmean':
                agg_grads=tr_mean(malicious_grads, n_attacker)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def min_max_median_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'
    multi_k = False
    candidates = []

    at_type='min-max'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_score(malicious_grads, agg_grads, n_attacker, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads,bulyan_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d | e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_sum_median_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_score(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        
        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='median'

    at_type='min-sum'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_score(malicious_grads, agg_grads, n_attacker, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads,bulyan_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d | e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def fang_mkrum_unknown_benign(nepochs=1200, fed_lr=0.5):
    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='fang'
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0
        
        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(user_grads[:n_attacker], agg_grads, deviation, n_attacker_)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_median(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)
            
            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()
                
            multi_k = True if aggregation == 'mkrum' else False
            if epoch_num == 0: print('multi krum is ', multi_k)
            agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            epoch_num+=1

def lie_mkrum_unknown_benign(nepochs=1200, fed_lr=0.5):
    def lie_attack(all_updates, z):
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='lie'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(user_grads[:n_attacker], agg_grads, deviation, n_attacker_)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_mkrum(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)

            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()
                
            multi_k = True if aggregation == 'mkrum' else False
            if epoch_num == 0: print('multi krum is ', multi_k)
            agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def agr_tailored_mkrum_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_mkrum(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([3.0]).cuda()

        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)
            if np.sum(krum_candidate < n_attackers) == n_attackers:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='our-agr'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation


    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(user_grads[:n_attacker], agg_grads, deviation, n_attacker_)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_mkrum(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)

            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()
                
            multi_k = True if aggregation == 'mkrum' else False
            if epoch_num == 0: print('multi krum is ', multi_k)
            agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_max_mkrum_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='min-max'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(user_grads[:n_attacker], agg_grads, deviation, n_attacker_)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_median(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)

            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()

            multi_k = True if aggregation == 'mkrum' else False
            if epoch_num == 0: print('multi krum is ', multi_k)
            agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_sum_mkrum_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_score(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        
        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='min-sum'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                n_attacker_ = max(1, n_attacker**2//nusers)
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(user_grads[:n_attacker], agg_grads, deviation, n_attacker_)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_median(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)

            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()

            multi_k = True if aggregation == 'mkrum' else False
            if epoch_num == 0: print('multi krum is ', multi_k)
            agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def fang_mkrum_known_benign(nepochs=1200, fed_lr=0.5):
    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='fang'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'paf':
                    malicious_grads=get_malicious_predictions_poison_all_far_sign(malicious_grads,nusers,n_attacker)
                elif at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_krum(malicious_grads, agg_grads, n_attacker, compression, q_level, norm)

            if not epoch_num:
                print(malicious_grads.shape)
                
            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':

                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0:
                    print('multi krum is ', multi_k)

                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads,bulyan_candidate=bulyan(malicious_grads, n_attacker)

                print(np.sum(bulyan_candidate<n_attacker))

                if n_attacker:
                    if epoch_num > 0 and (epoch_num%50==0 or epoch_num == (nepochs-1)):
                        try:
                            print('number of malicious grads chosen are ', np.array(candidates).reshape(5, 10))
                        except:
                            print('number of malicious grads chosen are ', np.array(candidates))
                        candidates = []
                        candidates.append(np.sum(bulyan_candidate<n_attacker))
                    else:
                        candidates.append(np.sum(bulyan_candidate<n_attacker))

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%1==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            epoch_num+=1

def lie_mkrum_known_benign(nepochs=1200, fed_lr=0.5):
    def lie_attack(all_updates, z):
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='LIE'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                    malicious_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_krum(malicious_grads, agg_grads, n_attacker, compression=compression, q_level=q_level, norm=norm)

            if not epoch_num : 
                print(malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='trmean':
                agg_grads=tr_mean(malicious_grads, n_attacker)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def agr_tailored_mkrum_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_mkrum(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([3.0]).cuda()

        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)
            if np.sum(krum_candidate < n_attackers) == n_attackers:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        return mal_updates

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='our-agr'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation


    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_mkrum(malicious_grads, agg_grads, n_attacker, dev_type=dev_type)

            if not malicious_grads.shape[0]==50: print('malicious grads shape ', malicious_grads.shape)
            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':

                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%1==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))
            
            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_max_mkrum_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='min-max'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_score(malicious_grads, agg_grads, n_attacker, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%1==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_sum_mkrum_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_score(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        
        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update
    
    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='mkrum'
    multi_k = False
    candidates = []

    at_type='min-sum'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_score(malicious_grads, agg_grads, n_attacker, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%1==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d n_mal_sel %d e %d | val loss %.4f val acc %.4f best val_acc %f'%(aggregation, at_type, n_attacker, np.sum(krum_candidate < n_attacker), epoch_num, val_loss, val_acc, best_global_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def fang_trmean_unknown_benign(nepochs=1200, fed_lr=0.5):
    def fang_attack_trmean_partial(all_updates, n_attackers):

        model_re = torch.mean(all_updates, 0)
        model_std = torch.std(all_updates, 0)
        deviation = torch.sign(model_re)
        
        max_vector_low = model_re + 3 * model_std 
        max_vector_hig = model_re + 4 * model_std
        min_vector_low = model_re - 4 * model_std
        min_vector_hig = model_re - 3 * model_std

        max_range = torch.cat((max_vector_low[:,None], max_vector_hig[:,None]), dim=1)
        min_range = torch.cat((min_vector_low[:,None], min_vector_hig[:,None]), dim=1)

        rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

        max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
        min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

        mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
            [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

        return mal_vec

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'
    multi_k = False
    candidates = []

    at_type='fang'
    z_values=[0.0]
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grad = get_malicious_updates_lie(user_grads[:n_attacker], n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    mal_updates = fang_attack_trmean_partial(user_grads[:n_attacker], n_attacker)

            malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]))
            
            if not epoch_num : 
                print(malicious_grads.shape)

            agg_grads=tr_mean(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def lie_trmean_unknown_benign(nepochs=1200, fed_lr=0.5):
    def lie_attack(all_updates, z):
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'
    multi_k = False
    candidates = []

    at_type='LIE'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                    mal_updates = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    mal_updates = fang_attack_trmean_partial(user_grads[:n_attacker], n_attacker)

            malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]))
            
            if not epoch_num : 
                print(malicious_grads.shape)
            
            agg_grads=tr_mean(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def agr_tailored_trmean_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_trmean(all_updates, model_re, n_attackers, dev_type='unit_vec', threshold=30):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([threshold]).cuda() #compute_lambda_our(all_updates, model_re, n_attackers)
        # print(lamda)
        threshold_diff = 1e-5
        prev_loss = -1
        lamda_fail = lamda
        lamda_succ = 0
        iters = 0 
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads = tr_mean(mal_updates, n_attackers)
            
            loss = torch.norm(agg_grads - model_re)
            
            if prev_loss < loss:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
            prev_loss = loss
            
        mal_update = (model_re - lamda_succ * deviation)
        return mal_update
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        return mal_updates

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'
    multi_k = False
    candidates = []

    at_type='our-agr'
    dev_type = 'std'
    threshold=10
    n_attackers=[10]
    partial_attackers = {4:1, 5:1, 8:2, 10:3, 12:4}

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                    mal_updates = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'our-agr':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_trmean(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                    mal_updates = torch.stack([mal_update]*n_attacker)
                    
            malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]))
            
            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()

            agg_grads=tr_mean(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def min_max_trmean_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec', threshold=30):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([threshold]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'
    multi_k = False
    candidates = []

    at_type='min-max'
    dev_type ='std'
    threshold=10
    n_attackers=[10]
    partial_attackers = {4:1, 5:1, 8:2, 10:3, 12:4}

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                elif at_type == 'fang':
                    mal_updates = fang_attack_trmean_partial(user_grads[:n_attacker], n_attacker)
                elif at_type == 'our-agr':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_median(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                elif at_type == 'min-max':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                elif at_type == 'min-sum':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)

            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()

            agg_grads=tr_mean(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%25==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d | e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_sum_trmean_unknown_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_score(all_updates, model_re, n_attackers, dev_type='unit_vec', threshold=30):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        
        lamda = torch.Tensor([threshold]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'

    at_type='min-sum'
    dev_type ='std'
    threshold=10
    n_attackers=[10]
    partial_attackers = {4:1, 5:1, 8:2, 10:3, 12:4}

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(user_grads[:n_attacker], z_values[n_attacker])
                    attacker_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    attacker_grads = fang_attack_trmean_partial(user_grads[:n_attacker], n_attacker)
                elif at_type == 'our-agr':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_median(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                elif at_type == 'min-max':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_dist(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)
                elif at_type == 'min-sum':
                    n_attacker_ = partial_attackers[n_attacker]
                    agg_grads = torch.mean(user_grads[:n_attacker], 0)
                    mal_update = our_attack_score(user_grads[:n_attacker], agg_grads, n_attacker_, threshold=threshold, dev_type=dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)

            if not (malicious_grads.shape[0]==50):
                print(malicious_grads.shape)
                sys.exit()

            agg_grads=tr_mean(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d | e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def fang_trmean_known_benign(nepochs=1200, fed_lr=0.5):
    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'
    multi_k = False
    candidates = []

    at_type='fang'
    z_values=[0.0]
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        for z in z_values:
            fed_file='alexnet_checkpoint_%s_%s_%d_%.2f.pth.tar'%(aggregation,at_type,n_attacker,z)
            fed_best_file='alexnet_best_%s_%s_%d_%.2f.pth.tar'%(aggregation,at_type,n_attacker,z)

            if resume:
                fed_checkpoint = chkpt+'/'+fed_file
                assert os.path.isfile(fed_checkpoint), 'Error: no user checkpoint at %s'%(fed_checkpoint)
                checkpoint = torch.load(fed_checkpoint, map_location='cuda:%d'%torch.cuda.current_device())
                fed_model.load_state_dict(checkpoint['state_dict'])
                optimizer_fed.load_state_dict(checkpoint['optimizer'])
                resume = 0
                best_global_acc=checkpoint['best_acc']
                best_global_te_acc=checkpoint['best_te_acc']
                val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
                epoch_num += checkpoint['epoch']
                print('resuming from epoch %d | val acc %.4f | best acc %.3f | best te acc %.3f'%(epoch_num, val_acc, best_global_acc, best_global_te_acc))

            torch.cuda.empty_cache()
            r=np.arange(user_tr_len)

            fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
            optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

            while epoch_num <= nepochs:
                user_grads=[]
                if not epoch_num and epoch_num%nbatches == 0:
                    np.random.shuffle(r)
                    for i in range(nusers):
                        user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                        user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

                for i in range(n_attacker, nusers):

                    inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                    targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                    inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                    outputs = fed_model(inputs)
                    loss = criterion(outputs, targets)
                    fed_model.zero_grad()
                    loss.backward(retain_graph=True)

                    param_grad=[]
                    for param in fed_model.parameters():
                        param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                    user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

                malicious_grads = user_grads

                if epoch_num in schedule:
                    for param_group in optimizer_fed.param_groups:
                        param_group['lr'] *= gamma
                        print('New learnin rate ', param_group['lr'])

                if n_attacker > 0:
                    if at_type == 'paf':
                        malicious_grads=get_malicious_predictions_poison_all_far_sign(malicious_grads,nusers,n_attacker)
                    elif at_type == 'lie':
                        malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                    elif at_type == 'fang':
                        agg_grads = torch.mean(malicious_grads, 0)
                        deviation = torch.sign(agg_grads)
                        malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
                    elif at_type == 'our':
                        agg_grads = torch.mean(malicious_grads, 0)
                        malicious_grads = our_attack_krum(malicious_grads, agg_grads, n_attacker, compression=compression, q_level=q_level, norm=norm)

                if not epoch_num : 
                    print(malicious_grads.shape)
                    
                if aggregation=='median':
                    agg_grads=torch.median(malicious_grads,dim=0)[0]

                elif aggregation=='average':
                    agg_grads=torch.mean(malicious_grads,dim=0)

                elif aggregation=='trmean':
                    agg_grads=tr_mean(malicious_grads, n_attacker)

                elif aggregation=='krum' or aggregation=='mkrum':
                    multi_k = True if aggregation == 'mkrum' else False
                    if epoch_num == 0: print('multi krum is ', multi_k)
                    agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)
                    
                elif aggregation=='bulyan':
                    agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

                del user_grads

                start_idx=0

                optimizer_fed.zero_grad()

                model_grads=[]

                for i, param in enumerate(fed_model.parameters()):
                    param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                    start_idx=start_idx+len(param.data.view(-1))
                    param_=param_.cuda()
                    model_grads.append(param_)

                optimizer_fed.step(model_grads)

                val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
                te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

                is_best = best_global_acc < val_acc

                best_global_acc = max(best_global_acc, val_acc)

                if is_best:
                    best_global_te_acc = te_acc

                if epoch_num%10==0 or epoch_num==nepochs-1:
                    print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

                if val_loss > 10:
                    print('val loss %f too high'%val_loss)
                    break
                
                epoch_num+=1

def lie_trmean_known_benign(nepochs=1200, fed_lr=0.5):
    def lie_attack(all_updates, z):
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'
    multi_k = False
    candidates = []

    at_type='LIE'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                    malicious_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_krum(malicious_grads, agg_grads, n_attacker, compression=compression, q_level=q_level, norm=norm)

            if not epoch_num : 
                print(malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='trmean':
                agg_grads=tr_mean(malicious_grads, n_attacker)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 10:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def agr_tailored_trmean_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_trmean(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).cuda() #compute_lambda_our(all_updates, model_re, n_attackers)
        # print(lamda)
        threshold_diff = 1e-5
        prev_loss = -1
        lamda_fail = lamda
        lamda_succ = 0
        iters = 0 
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads = tr_mean(mal_updates, n_attackers)
            
            loss = torch.norm(agg_grads - model_re)
            
            if prev_loss < loss:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
            prev_loss = loss
            
        mal_update = (model_re - lamda_succ * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        return mal_updates

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'
    multi_k = False
    candidates = []

    at_type='our-agr'
    dev_type = 'std'
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                    malicious_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    malicious_grads = our_attack_trmean(malicious_grads, agg_grads, n_attacker, dev_type=dev_type)

            if not epoch_num : 
                print(malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='trmean':
                agg_grads=tr_mean(malicious_grads, n_attacker)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break

            epoch_num+=1

def min_max_trmean_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'
    multi_k = False
    candidates = []

    at_type='min-max'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_score(malicious_grads, agg_grads, n_attacker, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='trmean':
                agg_grads=tr_mean(malicious_grads, n_attacker)
                
            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads,bulyan_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d | e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1

def min_sum_trmean_known_benign(nepochs=1200, fed_lr=0.5):
    def our_attack_score(all_updates, model_re, n_attackers, dev_type='unit_vec'):

        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        
        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update

    batch_size=250
    resume=0
    schedule=[1000]
    nbatches = user_tr_len//batch_size

    gamma=.5
    opt = 'sgd'
    criterion=nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    aggregation='trmean'
    multi_k = False
    candidates = []

    at_type='min-sum'
    dev_type ='std'
    z=0
    n_attackers=[10]

    arch='alexnet'
    chkpt='./'+aggregation

    for n_attacker in n_attackers:
        candidates = []

        epoch_num = 0
        best_global_acc = 0
        best_global_te_acc = 0

        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

        torch.cuda.empty_cache()
        r=np.arange(user_tr_len)

        while epoch_num <= nepochs:
            user_grads=[]
            if not epoch_num and epoch_num%nbatches == 0:
                np.random.shuffle(r)
                for i in range(nusers):
                    user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                    user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

            for i in range(n_attacker, nusers):

                inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)
                loss = criterion(outputs, targets)
                fed_model.zero_grad()
                loss.backward(retain_graph=True)

                param_grad=[]
                for param in fed_model.parameters():
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

            malicious_grads = user_grads

            if epoch_num in schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma
                    print('New learnin rate ', param_group['lr'])

            if n_attacker > 0:
                if at_type == 'lie':
                    malicious_grads = get_malicious_updates_lie(malicious_grads, n_attacker, z, epoch_num)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang(malicious_grads, agg_grads, deviation, n_attacker)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_score(malicious_grads, agg_grads, n_attacker, dev_type)

                mal_updates = torch.stack([mal_update] * n_attacker)
                malicious_grads = torch.cat((mal_updates, user_grads), 0)

            if epoch_num==0: print('malicious_grads shape ', malicious_grads.shape)

            if aggregation=='median':
                agg_grads=torch.median(malicious_grads,dim=0)[0]

            elif aggregation=='trmean':
                agg_grads=tr_mean(malicious_grads, n_attacker)
                
            elif aggregation=='average':
                agg_grads=torch.mean(malicious_grads,dim=0)

            elif aggregation=='krum' or aggregation=='mkrum':
                multi_k = True if aggregation == 'mkrum' else False
                if epoch_num == 0: print('multi krum is ', multi_k)
                agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

            elif aggregation=='bulyan':
                agg_grads,bulyan_candidate=bulyan(malicious_grads, n_attacker)

            del user_grads

            start_idx=0

            optimizer_fed.zero_grad()

            model_grads=[]

            for i, param in enumerate(fed_model.parameters()):
                param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx=start_idx+len(param.data.view(-1))
                param_=param_.cuda()
                model_grads.append(param_)

            optimizer_fed.step(model_grads)

            val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
            te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc

            best_global_acc = max(best_global_acc, val_acc)

            if is_best:
                best_global_te_acc = te_acc

            if epoch_num%10==0 or epoch_num==nepochs-1:
                print('%s: at %s n_at %d | e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

            if val_loss > 1000:
                print('val loss %f too high'%val_loss)
                break
                
            epoch_num+=1


if __name__ == "__main__":
    # load the train dataset
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar10_train = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=train_transform)

    cifar10_test = datasets.CIFAR10(root=data_loc, train=False, download=True, transform=train_transform)

    X=[]
    Y=[]
    for i in range(len(cifar10_train)):
        X.append(cifar10_train[i][0].numpy())
        Y.append(cifar10_train[i][1])

    for i in range(len(cifar10_test)):
        X.append(cifar10_test[i][0].numpy())
        Y.append(cifar10_test[i][1])

    X=np.array(X)
    Y=np.array(Y)

    print('total data len: ',len(X))

    if not os.path.isfile('./cifar10_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices,open('./cifar10_shuffle.pkl','wb'))
    else:
        all_indices=pickle.load(open('./cifar10_shuffle.pkl','rb'))

    X=X[all_indices]
    Y=Y[all_indices]


    # ## Divide cifar10 data among 50 clients in IID fashion

    # In[3]:


    # data loading

    nusers=50
    user_tr_len=1000

    total_tr_len=user_tr_len*nusers
    val_len=5000
    te_len=5000

    print('total data len: ',len(X))

    if not os.path.isfile('./cifar10_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices,open('./cifar10_shuffle.pkl','wb'))
    else:
        all_indices=pickle.load(open('./cifar10_shuffle.pkl','rb'))

    total_tr_data=X[:total_tr_len]
    total_tr_label=Y[:total_tr_len]

    val_data=X[total_tr_len:(total_tr_len+val_len)]
    val_label=Y[total_tr_len:(total_tr_len+val_len)]

    te_data=X[(total_tr_len+val_len):(total_tr_len+val_len+te_len)]
    te_label=Y[(total_tr_len+val_len):(total_tr_len+val_len+te_len)]

    total_tr_data_tensor=torch.from_numpy(total_tr_data).type(torch.FloatTensor)
    total_tr_label_tensor=torch.from_numpy(total_tr_label).type(torch.LongTensor)

    val_data_tensor=torch.from_numpy(val_data).type(torch.FloatTensor)
    val_label_tensor=torch.from_numpy(val_label).type(torch.LongTensor)

    te_data_tensor=torch.from_numpy(te_data).type(torch.FloatTensor)
    te_label_tensor=torch.from_numpy(te_label).type(torch.LongTensor)

    print('total tr len %d | val len %d | test len %d'%(len(total_tr_data_tensor),len(val_data_tensor),len(te_data_tensor)))

    #==============================================================================================================

    user_tr_data_tensors=[]
    user_tr_label_tensors=[]

    for i in range(nusers):
        
        user_tr_data_tensor=torch.from_numpy(total_tr_data[user_tr_len*i:user_tr_len*(i+1)]).type(torch.FloatTensor)
        user_tr_label_tensor=torch.from_numpy(total_tr_label[user_tr_len*i:user_tr_len*(i+1)]).type(torch.LongTensor)

        user_tr_data_tensors.append(user_tr_data_tensor)
        user_tr_label_tensors.append(user_tr_label_tensor)
        print('user %d tr len %d'%(i,len(user_tr_data_tensor)), flush=True)
    
    # get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--aggregation", type=str, help = "Select aggregation")
    parser.add_argument("--attack", type=str, help = "Select attack")
    parser.add_argument("--benign_known", type=str, help = "Gradients of benign devices are known or unknown")
    parser.add_argument("--epoch", type=int, help = "Number of epochs")
    parser.add_argument("--learning_rate", type=float, help = "Federated learning rate")

    benign_known = False
    aggregation = "bulyan"
    attack = "fang"
    epoch = 1200
    learning_rate = 0.5

    args = parser.parse_args()
    print(args)
    if args.benign_known:
        benign_known = True if args.benign_known == "known" else False
    if args.attack:
        attack = args.attack
    if args.aggregation:
        aggregation = args.aggregation
    if args.epoch:
        epoch = args.epoch
    if args.learning_rate:
        learning_rate = args.learning_rate

    if benign_known:
        if aggregation == "bulyan":
            if attack == "fang":
                fang_bulyan_known_benign(epoch, learning_rate)
            if attack == "lie":
                lie_bulyan_known_benign(epoch, learning_rate)
            if attack == "agr-tailored":
                agr_tailored_bulyan_known_benign(epoch, learning_rate)
            if attack == "min-max":
                min_max_bulyan_known_benign(epoch, learning_rate)
            if attack == "min-sum":
                min_sum_bulyan_known_benign(epoch, learning_rate)
        if aggregation == "median":
            if attack == "fang":
                fang_median_known_benign(epoch, learning_rate)
            if attack == "lie":
                lie_median_known_benign(epoch, learning_rate)
            if attack == "agr-tailored":
                agr_tailored_median_known_benign(epoch, learning_rate)
            if attack == "min-max":
                min_max_median_known_benign(epoch, learning_rate)
            if attack == "min-sum":
                min_sum_median_known_benign(epoch, learning_rate)
        if aggregation == "multi-krum":
            if attack == "fang":
                fang_mkrum_known_benign(epoch, learning_rate)
            if attack == "lie":
                lie_mkrum_known_benign(epoch, learning_rate)
            if attack == "agr-tailored":
                agr_tailored_mkrum_known_benign(epoch, learning_rate)
            if attack == "min-max":
                min_max_mkrum_known_benign(epoch, learning_rate)
            if attack == "min-sum":
                min_sum_mkrum_known_benign(epoch, learning_rate)
        if aggregation == "trmean":
            if attack == "fang":
                fang_trmean_known_benign(epoch, learning_rate)
            if attack == "lie":
                lie_trmean_known_benign(epoch, learning_rate)
            if attack == "agr-tailored":
                agr_tailored_trmean_known_benign(epoch, learning_rate)
            if attack == "min-max":
                min_max_trmean_known_benign(epoch, learning_rate)
            if attack == "min-sum":
                min_sum_trmean_known_benign(epoch, learning_rate)
    else:
        if aggregation == "bulyan":
            if attack == "fang":
                fang_bulyan_unknown_benign(epoch, learning_rate)
            if attack == "lie":
                lie_bulyan_unknown_benign(epoch, learning_rate)
            if attack == "agr-tailored":
                agr_tailored_bulyan_unknown_benign(epoch, learning_rate)
            if attack == "min-max":
                min_max_bulyan_unknown_benign(epoch, learning_rate)
            if attack == "min-sum":
                min_sum_bulyan_unknown_benign(epoch, learning_rate)
        if aggregation == "median":
            if attack == "fang":
                fang_median_unknown_benign(epoch, learning_rate)
            if attack == "lie":
                lie_median_unknown_benign(epoch, learning_rate)
            if attack == "agr-tailored":
                agr_tailored_median_unknown_benign(epoch, learning_rate)
            if attack == "min-max":
                min_max_median_unknown_benign(epoch, learning_rate)
            if attack == "min-sum":
                min_sum_median_unknown_benign(epoch, learning_rate)
        if aggregation == "multi-krum":
            if attack == "fang":
                fang_mkrum_unknown_benign(epoch, learning_rate)
            if attack == "lie":
                lie_mkrum_unknown_benign(epoch, learning_rate)
            if attack == "agr-tailored":
                agr_tailored_mkrum_unknown_benign(epoch, learning_rate)
            if attack == "min-max":
                min_max_mkrum_unknown_benign(epoch, learning_rate)
            if attack == "min-sum":
                min_sum_mkrum_unknown_benign(epoch, learning_rate)
        if aggregation == "trmean":
            if attack == "fang":
                fang_trmean_unknown_benign(epoch, learning_rate)
            if attack == "lie":
                lie_trmean_unknown_benign(epoch, learning_rate)
            if attack == "agr-tailored":
                agr_tailored_trmean_unknown_benign(epoch, learning_rate)
            if attack == "min-max":
                min_max_trmean_unknown_benign(epoch, learning_rate)
            if attack == "min-sum":
                min_sum_trmean_unknown_benign(epoch, learning_rate)