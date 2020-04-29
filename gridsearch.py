from training import train
from training import accuracy
from resnetclass import get_resnet
import numpy as np
import torch


def gridsearch(trainset, testset,loss_fn,precision=.2,epsilon = .2,phi=1, bs=64,learning_rate=1e-2,n_epoch=2):
  solu=1,1,1
  depth, width, resol = [2,2,2,2], 16, 224  ## ce sont les valeurs de base à scaler ensuite
  # training
  model1 = get_resnet(depth, resol, width)
  optimizer = torch.optim.Adam(model1.parameters(),lr=learning_rate)
  model1 =train(model1, optimizer, trainset, loss_fn, batch_size=bs, n_epoch=n_epoch, disp_stats=False, validate=False)
  # testing
  accu_max=accuracy(model1, testset, bs)

  for alpha in range(1,2,precision):
    for beta in range(1,2,precision):
      for gamma in range(1,2,precision):
        if np.abs(alpha*beta**2*gamma**2 - 2)<=epsilon:
          # il faut réentraîner le modele
          new_depth = [alpha*d for d in depth]
          model = get_resnet(new_depth, beta*width, gamma*resol)
          optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
          model = train(model, optimizer, trainset, loss_fn, batch_size=bs, n_epoch=n_epoch, disp_stats=False, validate=False)
          accu = accuracy(model, testset, batch_size = bs)
          if accu>accu_max:
            accu_max= accu
            solu=alpha,beta,gamma
    return solu

