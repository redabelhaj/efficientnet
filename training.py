from get_data import get_loader
import torch
import numpy as np
import torch.nn.functional as F
from resnetclass import get_resnet
import tqdm


def interpolate(img, res):
    if img.ndim ==4:
        img_perm = F.interpolate(img, size = res)
        img_perm = img_perm.permute(0,1,3,2)
        img_perm = F.interpolate(img_perm, size =res)
        img_perm = img_perm.permute(0,1,3,2)
        return img_perm
    elif img.ndim==3:
        img_perm = F.interpolate(img, size = res)
        img_perm = img_perm.permute(0,2,1)
        img_perm = F.interpolate(img_perm, size =res)
        img_perm = img_perm.permute(0,2,1)
        return img_perm
    else:
        raise Exception("tensor dimension should be 3 or 4")



def accuracy(net,testset,num_points = 9000,batch_size = 64,cuda=True):
    r=net.resolution
    testloader = get_loader(testset, batch_size, num_points=num_points)
    ## les images sont sous formes de dataloader de tenseurs d'une certaine taille
    net.eval()
    correct = 0
    total = 0
    if cuda:
        net.to('cuda')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images2 = interpolate(images,r) 
            if cuda:
                images2= images2.to('cuda')
                labels = labels.to('cuda')
            ## dans tous les cas ici il faut changer la resol de l'image 
            outputs = net(images2)
            _, predicted = torch.max(outputs.data, 1)
           
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 100.0 * correct/total




def train(model, optimizer, trainset, loss_fn,num_points=5000, batch_size=220,  n_epoch = 40, cuda=True,disp=True):
    if cuda:
        model.to('cuda')
    resol_cible = model.resolution
    trainloader = get_loader(trainset, batch_size, num_points=num_points)
    loss_func = loss_fn()
    if disp==True:
        for _ in tqdm.notebook.tqdm(range(n_epoch)):
            #print('epoch', 1+i, '/', n_epoch)
            for data in trainloader:
                inputs, labels = data
                inputs2 = interpolate(inputs, resol_cible)
                if cuda:
                    inputs2 = inputs2.to('cuda') 
                    labels = labels.to('cuda')
                optimizer.zero_grad()
                outputs = model(inputs2)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
        return model
    elif disp==False:
        for _ in range(n_epoch):
            #print('epoch', 1+i, '/', n_epoch)
            for data in trainloader:
                inputs, labels = data
                inputs2 = interpolate(inputs, resol_cible)
                if cuda:
                    inputs2 = inputs2.to('cuda') 
                    labels = labels.to('cuda')
                optimizer.zero_grad()
                outputs = model(inputs2)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
        return model
        

def train_acc_once(trainset, testset,num_points=5000, batch_size=220,width=16,resolution = 110, depth = [2,2,2,2],n_epoch=30, cuda=True,disp=True):
    model = get_resnet(depth = depth, resolution=resolution, width=width)
    model=model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss
    model = train(model, optimizer, trainset, loss_fn, num_points=num_points, batch_size=batch_size, n_epoch=n_epoch, cuda=cuda,disp=disp)
    acc = accuracy(model, testset, batch_size=batch_size)
    del model
    return acc

def train_acc_crossval(trainset, testset,num_points=5000, batch_size=220,width=16,resolution = 110, depth = [2,2,2,2],n_epoch=30, cuda=True,k=30,disp=True):
    accu_tot = 0
    for i in range(k):
        i+=1
        #print('validation',i,'/',k)
        acc = train_acc_once(trainset, testset, num_points=num_points, batch_size=batch_size, width=width, resolution=resolution, depth = depth, n_epoch=n_epoch, cuda = cuda,disp=disp)
        accu_tot+=acc
        print('accu obtenue:', acc)
    return accu_tot/k

def train_acc_multiple(trainset, testset,num_points=5000, batch_size=220,width=16,resolution = 110, depth = [2,2,2,2],n_epoch=30, cuda=True,k=30):
    list_acc = []
    for i in range(k):
        i+=1
        print('training',i,'/',k)
        acc = train_acc_once(trainset, testset, num_points=num_points, batch_size=batch_size, width=width, resolution=resolution, depth = depth, n_epoch=n_epoch, cuda = cuda)
        list_acc.append(acc)
        print('accu obtenue:', acc)
    return list_acc


def train_w_loss(model, optimizer, trainset, loss_fn,num_points=5000, batch_size=220,  n_epoch = 40):
    model = model.to('cuda')
    resol_cible = model.resolution
    trainloader = get_loader(trainset, batch_size, num_points=num_points)
    loss_func = loss_fn()
    
    losses = []
    for i in tqdm.notebook.tqdm(range(n_epoch)):
        if i%10==0:
            print('epoch', 1+i ,"/",n_epoch )
        rloss=0
        #print('epoch', 1+i, '/', n_epoch)
        for data in trainloader:
            inputs, labels = data
            inputs2 = interpolate(inputs, resol_cible)
            inputs2 = inputs2.to('cuda') 
            labels = labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs2)
            loss = loss_func(outputs, labels)
            rloss+=loss
            loss.backward()
            optimizer.step()
        losses.append(rloss)
    return model, losses


