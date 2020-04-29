from training import train
from training import accuracy
from resnetclass import get_resnet
import numpy as np
import torch


def gridsearch(trainset, testset,loss_fn,precision,epsilon,phi=1, bs=64,learning_rate=1e-2,n_epoch=2):
  ## il faut d'abord entraîner les modèles avant de trouver l'accuracy !!
  solu=1,1,1
  depth, width, resol = [2,2,2,2], 16, 224  ## ce sont les valeurs de base à scaler ensuite
  # training
  model1 = get_resnet(depth, resol, width)
  optimizer = torch.optim.Adam(model1.parameters(),lr=learning_rate)
  model1 = train(model1, optimizer,trainset,loss_fn,bs, n_epoch, disp_stats=False)
  # testing
  accu_max=accuracy(model1, testset, bs)

  for alpha in range(1,2,precision):
    for beta in range(1,2,precision):
      for gama in range(1,2,precision):
        if np.abs(alpha*beta**2*gama**2 - 2)<=epsilon:
          # il faut réentraîner le modele
          new_depth = [alpha*d for d in depth]
          model = get_resnet(new_depth, beta*width, gama*resol)
          optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
          model = train(model, optimizer, trainset, loss_fn, bs, n_epoch, disp_stats=False)
          accu = accuracy(model, testset, bs)
          if accu>accu_max:
            accu_max= accu
            solu=alpha,beta,gama
    return solu

