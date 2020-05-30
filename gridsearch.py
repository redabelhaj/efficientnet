import numpy as np
import torch
from resnetclass import get_resnet
from training import train_acc_crossval


def get_new_depth(alpha, depth):
    sd = sum(depth)
    i=0
    new_depth = depth.copy()
    while(sum(new_depth)<alpha*sd):
        new_depth[i%len(depth)] +=1
        i+=1
    return new_depth

def get_list_gs(precision=.12, epsilon=0.12, max_flops=2):
    return [(round(a,2),round(b,2),round(g,2)) for a in np.arange(1.13,2,precision) for b in np.arange(1.09,2,precision)
            for g in np.arange(1.05,2,precision) if np.abs(a*(b**2)*(g**2) - max_flops)< epsilon ]

def grid_search(list_abg, max_ref,trainset, testset,disp=False):
    solu =1,1,1
    best = max_ref
    for a,b,g in list_abg:
        res=int(95*g)
        w=int(12*b)
        d=get_new_depth(a,[2,2,2,2])
        accu = train_acc_crossval(trainset, testset,num_points=7000, batch_size=220,
                           width=w,resolution = res, depth = d,n_epoch=40,k=8,disp=disp)
        print('params',a,b,g,' acc : ',accu)
        if accu>best:
            solu = a,b,g
            best = accu
    return solu