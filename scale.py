import numpy as np
from gridsearch import get_new_depth
import time as t 
from training import train_acc_multiple


def scale_depth(trainset, testset, n_pts= 4, max_flops=2, k=8):
    flops = np.linspace(1,max_flops, n_pts)
    res = []
    for flop in flops:
        new_depth = get_new_depth(flop, [2,2,2,2])
        accus = train_acc_multiple(trainset, testset,num_points=7000, 
                           batch_size=220,width=12,resolution = 95, depth = new_depth,n_epoch=40,k=k)
        res.append(accus)
    return res

def scale_width(trainset, testset, n_pts= 4, max_flops=2, k=8):
    flops = np.linspace(1,max_flops, n_pts)
    res = []
    for flop in flops:
        new_width = int(np.sqrt(flop)*12)
        accus = train_acc_multiple(trainset, testset,num_points=7000, 
                           batch_size=220,width=new_width,resolution = 95, depth = [2,2,2,2],n_epoch=40,k=k)
        res.append(accus)
    return res


def scale_resol(trainset, testset, n_pts= 4, max_flops=2, k=8):
    flops = np.linspace(1,max_flops, n_pts)
    res = []
    for flop in flops:
        new_resol = int(np.sqrt(flop)*95)
        accus = train_acc_multiple(trainset, testset,num_points=7000, 
                           batch_size=220,width=12,resolution = new_resol, depth = [2,2,2,2],n_epoch=40,k=k)
        res.append(accus)
    return res