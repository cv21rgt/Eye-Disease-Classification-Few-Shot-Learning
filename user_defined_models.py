
import torch
import torch.nn as nn

class CNN_BLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNN_BLOCK, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)         
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()        
        self.pool= nn.MaxPool2d(kernel_size=(2, 2), stride=2) # This halves the image size

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class FeatureExtractor(nn.Module):

    def __init__(self, in_channels, device="cpu"):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
          CNN_BLOCK(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
          CNN_BLOCK(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
          CNN_BLOCK(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
          CNN_BLOCK(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),          
          nn.Flatten()
        )

    def forward(self, input):
      x = self.feature_extractor(input)
      return x

