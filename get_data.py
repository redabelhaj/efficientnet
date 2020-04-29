import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

def get_dataset(resol_max,train_size, test_size,path, validation = False, database = 'cifar10', download = True):

    transform = transforms.Compose([transforms.Resize(size=(resol_max, resol_max)), transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    if validation:
        raise NotImplementedError
    if database=='cifar10':
        trainset = torchvision.datasets.CIFAR10(root= path, train=True, transform=transform, download=download)
        testset = torchvision.datasets.CIFAR10(root = path, train=False,transform=transform, download=download)
    else:
        raise NotImplementedError
    return trainset, testset


def get_loader(data, batch_size, num_workers=2):
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True,num_workers = num_workers)
    return loader


