from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image as pil


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()  
        # in_channel = 1
        # First layer
        self.layer1 = nn.Sequential(nn.Conv2d(1,32,3,1,padding=1),
                                nn.ReLU(True),
                                nn.MaxPool2d(2,2))
        # Second layer
        self.layer2 = nn.Sequential(nn.Conv2d(32,32,3,1,padding=1),
                                nn.ReLU(True),
                                nn.MaxPool2d(2,2))
              
        # Third layer 
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(32,16,2,2),
                                nn.ReLU(True))

        # Forth layer 
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(16,8,2,2),
                                nn.ReLU(True))

        # Fifth layer 
        self.layer5 = nn.Sequential(nn.Conv2d(8,1,kernel_size=1),
                                nn.ReLU(True))
    def forward(self,x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)        
        conv4 = self.layer4(conv3)                         
        fc_out = self.layer5(conv4)
        return fc_out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,features=[16,32,64,128]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # The down part of Unet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # The up part of Unet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2,feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        connections = []
        for down in self.downs:
            x = down(x)
            connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        connections = connections[::-1]        
        for i in range(0, len(self.ups),2):
            x = self.ups[i](x)
            connection = connections[i//2]

            if x.shape != connection.shape:
                tran = transforms.Resize(size=connection.shape[2:])
                x = tran(x)
            concat = torch.cat((connection,x), dim=1)
            x = self.ups[i+1](concat)
        
        return self.final(x)
