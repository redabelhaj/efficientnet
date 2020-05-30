
import torch
import torch.nn as nn 
import torch.nn.functional as F




class ResNetBlock(nn.Module):
  def __init__(self, in_planes, planes, stride=1, bn=True):
      super(ResNetBlock, self).__init__()
      self.stride = stride
      self.in_planes=in_planes
      self.planes = planes
      if stride!=1:
        if not(bn):
          self.fx = nn.Sequential(nn.Conv2d(in_planes, planes, 3, stride=2, 
                                            padding=1),
                                  nn.ReLU(), 
                                  nn.Conv2d(planes, planes,3, padding=1))
          
          self.iden = nn.Conv2d(self.in_planes, self.planes, 3, stride = 2,
                                padding =1)
        else:
          self.fx = nn.Sequential(nn.Conv2d(in_planes, planes, 3, stride=2, 
                                            padding=1),
                                  nn.BatchNorm2d(planes),
                                  nn.ReLU(), 
                                  nn.Conv2d(planes, planes,3, padding=1),
                                  nn.BatchNorm2d(planes))
          
          self.iden = nn.Conv2d(self.in_planes, self.planes, 3, stride = 2,
                                padding =1)

      else:
        if not(bn):
          self.fx = nn.Sequential(nn.Conv2d(planes, planes, 3, padding = 1),
                                  nn.ReLU(), 
                                  nn.Conv2d(planes, planes,3, padding=1))
          self.iden = nn.Sequential()
        else:
          self.fx = nn.Sequential(nn.Conv2d(planes, planes, 3, padding = 1),
                                  nn.BatchNorm2d(planes),
                                  nn.ReLU(), 
                                  nn.Conv2d(planes, planes,3, padding=1),
                                  nn.BatchNorm2d(planes))
        
          self.iden = nn.Sequential()

          

  def forward(self, x):


    if self.stride ==1:
      fx = self.fx(x)
    
      
      out = fx + self.iden(x)
      return F.relu(out)

    else:

      fx = self.fx(x)
      out = fx + self.iden(x)
      return F.relu(out)




class ResNet(nn.Module):

  def __init__(self, depth,resolution,width=16, num_classes=10,input_dim=3,Block = ResNetBlock, bn = True):
      super(ResNet, self).__init__()
      self.in_planes = width
      self.conv1 = nn.Conv2d(input_dim, width, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(width)
      self.depth = depth
      self.width = width
      self.resolution = resolution
      
      resol = resolution
      # construction des piles
      plane = width
      layers = []
      for nb in depth:
        layer = self._make_layer(Block,plane ,nb,2,bn=bn)
        layers.append(layer)
        plane*=2
        resol= roundsp(resol)
      
      self.layers = nn.Sequential(*layers)
      # arrivé ici on en est à plane@resol*resol
      # on applique le pooling
      self.pool = nn.AvgPool2d(4)
      resol = int(resol/4)

      out_size = resol*resol*plane 
      self.linear = nn.Linear(out_size, num_classes)

  def _make_layer(self, Block, planes, num_blocks, stride, bn = True):
      layers = []
      block1 = Block(planes, 2*planes, stride = 2, bn=bn)
      planes *=2
      layers.append(block1)
      for _ in range(1,num_blocks):
        block_new = Block(planes, planes, stride =1, bn = bn)
        layers.append(block_new)
      return nn.Sequential(*layers)

  def forward(self, x):
      
    # premiere etape
      out = F.relu(self.bn1(self.conv1(x)))
     
      # passage par les piles 
      out = self.layers(out)
      
      # output
      out = self.pool(out)
      out = out.view(out.size(0), -1)
      out = self.linear(out)
      return out

def get_resnet(depth = [2,2,2,2],resolution = 224,width=16, num_classes=10,input_dim=3):
  resnet = ResNet(depth, resolution, width=width)

  return resnet

def roundsp(r):
  if r%2==0:
    return r/2
  else:
    return (1+r)/2