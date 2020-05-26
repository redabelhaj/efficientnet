import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

def get_dataset(resol_max,path, download = True):

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root= path, train=True, transform=transform, download=download)
    testset = torchvision.datasets.CIFAR10(root = path, train=False,transform=transform, download=download)
    return trainset, testset


def get_loader(data, batch_size, num_points = 10000, num_workers=2):
    idx = np.arange(num_points)
    smallset = torch.utils.data.Subset(data, idx)
    loader = torch.utils.data.DataLoader(smallset, batch_size=batch_size, shuffle=False,num_workers = num_workers)
    return loader


