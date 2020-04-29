from get_data import get_loader
import torch
import numpy as np
import torch.nn.functional as F

def accuracy(net,testset,batch_size = 64,cuda=True):
    r=net.resolution
    testloader = get_loader(testset, batch_size)
    net.eval()
    correct = 0
    total = 0
    if cuda:
        net.to('cuda')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if cuda:
                images = images.to('cuda')
                labels = labels.to('cuda')
            ## dans tous les cas ici il faut changer la resol de l'image 
            images = F.interpolate(images, r)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
           
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 100.0 * correct/total




def train(model, optimizer, trainset, loss_fn, valset=None, batch_size=64,  n_epoch = 5, cuda=True, disp_stats = True, validate = False):
    if cuda:
        model.to('cuda')

    if validate :
        valloader = get_loader(valset, batch_size)


    resol_cible = model.resolution
    trainloader = get_loader(trainset, batch_size)

    loss_train, loss_val = [], []
    acc_train, acc_val = [],[]

    loss_func = loss_fn()
    
    for epoch in range(n_epoch):
        running_trainloss = 0
        for data in trainloader:
            inputs, labels = data
            if cuda:
                inputs = inputs.to('cuda') 
                labels = labels.to('cuda')

            # on change la resol de l'image    
            optimizer.zero_grad()
            inputs = F.interpolate(inputs, resol_cible)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            running_trainloss+= loss
            optimizer.step()
    
        # maintenant plus qu'a calculer les stats et les afficher
        loss_train.append(running_trainloss)
        acct = accuracy(model, trainset, batch_size=batch_size)
        acc_train.append(acct)
        if disp_stats:
            print('Epoch', epoch, '/', n_epoch, ':')
            print('train loss : ', running_trainloss)
            print('train acc',acct )

        # stats de validation
        if validate:
            vloss = 0
            for data in valloader:
                ipt, lbl = data
                if cuda : 
                    ipt = ipt.to('cuda')
                    lbl = ipt.to('cuda')
                ipt = F.interpolate(ipt, resol_cible)
                out = model(ipt)
                vloss+= loss_func(out, lbl)
            loss_val.append(vloss)
            accv = accuracy(model, valset, batch_size)
            acc_val.append(accv)
            if disp_stats:
                print('val loss : ', vloss)
                print('val acc' ,accv)

    if validate:
        raise NotImplementedError
    else:
        return model, loss_train, acc_train


