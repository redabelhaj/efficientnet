from get_data import get_loader_from_data
import tqdm
import torch

def train(model, optimizer, trainset, loss_fn, batch_size=64, val_split = .10, n_epoch = 5, cuda=True, disp_stats = True):
    total_nb_ex = len(trainset)
    train_idx = int((1-val_split)*total_nb_ex)

    trainset = trainset[:train_idx]
    valset = trainset[train_idx:]

    resol_cible = model.resolution
    trainloader = get_loader_from_data(trainset, resol_cible, batch_size = batch_size)

    valloader = get_loader_from_data(valset, resol_cible, batch_size=batch_size)


    loss_train, loss_val = [], []
    #acc_train, acc_val = [],[] ## fct accuracy non encore codée

    loss_func = loss_fn()
    
    for epoch in tqdm.tqdm(range(n_epoch)):
        running_trainloss = 0
        for data in tqdm.tqdm(trainloader):
            inputs, labels = data
            if cuda:
                inputs = inputs.to('cuda') 
                labels = labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            running_trainloss+= loss
            optimizer.step()
    
        # maintenant plus qu'a calculer les stats et les afficher
        loss_train.append(running_trainloss)
        #acc_train = ## il faut la fonction accuracy

        vloss = 0
        for data in valloader:
            ipt, lbl = data
            if cuda : 
                ipt = ipt.to('cuda')
                lbl = ipt.to('cuda')
            out = model(ipt)

            vloss+= loss_func(out, lbl)
        loss_val.append(vloss)
        print('Epoch', epoch, '/', n_epoch, ':')
        print('train loss : ', running_trainloss)
        print('val loss : ', vloss)
        # print('train acc' ) etc....

# a modifier s'il faut retourner les accuracys
    return model, loss_train, loss_val

