import numpy as np
import torch
from resnetclass import get_resnet
from training import accuracy
from training import train

def get_new_depth(alpha, depth):
  old_sum_depth = sum(depth)
  new_sum_depth = int(alpha*old_sum_depth)
  new_depth = depth 
  new_sum = old_sum_depth
  for i in range(len(depth)):
    if new_sum < new_sum_depth:
      new_depth[i]+=1
      new_sum +=1
  return new_depth


def gridsearch(trainset, testset,loss_fn,precision=.2,epsilon = .5,bs=64,n_epoch=2):
  solu=1,1,1
  depth, width, resol = [2,2,2,2], 16, 112  ## ce sont les valeurs de base à scaler ensuite
  # training
  model1 = get_resnet(depth, resol, width)
  optimizer = torch.optim.Adam(model1.parameters())
  model1, _, _ =train(model1, optimizer, trainset, loss_fn, batch_size=bs, n_epoch=n_epoch, disp_stats=False, validate=False)
  # testing
  accu_max=accuracy(model1, testset, batch_size=bs)

  for alpha in np.arange(1,2,precision):
    for beta in np.arange(1,2,precision):
      for gamma in np.arange(1,2,precision):
        if np.abs(alpha*beta**2*gamma**2 - 2)<=epsilon:
          # il faut réentraîner le modele
          new_depth = get_new_depth(alpha, depth)
          model = get_resnet(depth = new_depth, width = int(beta*width), resolution = int(gamma*resol))
          optimizer = torch.optim.Adam(model.parameters())
          model, _, _ = train(model, optimizer, trainset, loss_fn, batch_size=bs, n_epoch=n_epoch, disp_stats=False, validate=False)
          accu = accuracy(model, testset, batch_size = bs)
          if accu>accu_max:
            accu_max= accu
            solu=alpha,beta,gamma
  return solu