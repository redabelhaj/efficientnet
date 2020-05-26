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

def gridsearch(trainset, testset,loss_fn,num_points = 4000, precision=.2,epsilon = .5,bs=64,n_epoch=30, k=30):
  solu=1,1,1
  depth, width, resol = [2,2,2,2], 16, 112  ## ce sont les valeurs de base à scaler ensuite
  # training
  accu_max= train_acc_crossval(trainset, testset, num_points=num_points, batch_size=bs, width = width, resolution=resol, depth = [2,2,2,2],
  n_epoch=n_epoch, k=k)

  for alpha in np.arange(1,2,precision):
    for beta in np.arange(1,2,precision):
      for gamma in np.arange(1,2,precision):
        if np.abs(alpha*beta**2*gamma**2 - 2)<=epsilon:
          # il faut réentraîner le modele
          new_depth = get_new_depth(alpha, depth)
          new_width = int(beta*width)
          new_resol = int(gamma*resol)
          accu = train_acc_crossval(trainset, testset, num_points=num_points, batch_size=bs, width = new_width, 
          resolution=new_resol, depth = new_depth, n_epoch=n_epoch, k=k)
          
          if accu>accu_max:
            accu_max= accu
            solu=alpha,beta,gamma
  return solu