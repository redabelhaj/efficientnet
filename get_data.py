import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

def get_data(resol_max,train_size, test_size,path, database = 'cifar10', download = True):

    transform = transforms.Compose([transforms.Resize(size=(resol_max, resol_max)), transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    if database=='cifar10':
        trainset = torchvision.datasets.CIFAR10(root= path, train=True, transform=transform, download=download)
        testset = torchvision.datasets.CIFAR10(root = path, train=False,transform=transform, download=download)
    else:
        raise NotImplementedError
    return trainset, testset


def get_loader_from_data(data,resol_cible, batch_size, num_workers=2):

    transform = transforms.Resize(size = (resol_cible, resol_cible))
    data = transform(data)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True,num_workers = num_workers)
    return loader


