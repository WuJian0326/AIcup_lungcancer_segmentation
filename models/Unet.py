import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class PyramidPool(nn.Module):
  def __init__(self, in_features, out_features, pool_size):
    super(PyramidPool, self).__init__()
    self.features = nn.Sequential(
      nn.AdaptiveAvgPool2d(pool_size),
      nn.Conv2d(in_features, out_features, 1, bias=False),
      nn.BatchNorm2d(out_features),
      nn.LeakyReLU(inplace=True)
    )

  def forward(self, x):
    size = x.size()
    output = F.upsample(self.features(x), size[2:], mode='bilinear')
    return output


class ASSP(nn.Module):
  def __init__(self, in_channels, out_channels=256):
    super(ASSP, self).__init__()

    self.relu = nn.ReLU(inplace=True)

    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           padding=0,
                           dilation=1,
                           bias=False)

    self.bn1 = nn.BatchNorm2d(out_channels)

    self.conv2 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=6,
                           dilation=6,
                           bias=False)

    self.bn2 = nn.BatchNorm2d(out_channels)

    self.conv3 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=12,
                           dilation=12,
                           bias=False)

    self.bn3 = nn.BatchNorm2d(out_channels)

    self.conv4 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=18,
                           dilation=18,
                           bias=False)

    self.bn4 = nn.BatchNorm2d(out_channels)

    self.conv5 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           dilation=1,
                           bias=False)

    self.bn5 = nn.BatchNorm2d(out_channels)

    self.convf = nn.Conv2d(in_channels=out_channels * 5,
                           out_channels=out_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           dilation=1,
                           bias=False)

    self.bnf = nn.BatchNorm2d(out_channels)
    self.adapool = nn.AdaptiveAvgPool2d(1)

    self.PSP1 = PyramidPool(in_channels, out_channels, 1)
    self.PSP3 = PyramidPool(in_channels, out_channels, 3)

  def forward(self, x):
    x1 = self.conv1(x)
    x1 = self.bn1(x1)
    x1 = self.relu(x1)

    x2 = self.conv2(x)
    x2 = self.bn2(x2)
    x2 = self.relu(x2)

    x3 = self.conv3(x)
    x3 = self.bn3(x3)
    x3 = self.relu(x3)

    x4 = self.conv4(x)
    x4 = self.bn4(x4)
    x4 = self.relu(x4)

    x5 = self.adapool(x)
    x5 = self.conv5(x5)
    x5 = self.relu(x5)
    x5 = F.interpolate(x5, size=tuple(x4.shape[-2:]), mode='bilinear')

    x = torch.cat((x1, x2, x3, x4, x5), dim=1)
    x = self.convf(x)
    x = self.bnf(x)
    x = self.relu(x)

    return x

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(inplace=True),
    )
  def forward(self, x):
    return self.conv(x)

class ResNet(nn.Module):
    def __init__(self, in_channels=3, conv1_out=64):
      super(ResNet, self).__init__()
      self.resnet = models.resnet34(pretrained=True)
      self.conv1 = nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3),bias=False)
      self.conv2 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)
      self.conv3 = nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),bias=False)
      self.conv4 = nn.Conv2d(3, 16, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), bias=False)
      self.relu = nn.LeakyReLU(inplace=True)

      self.assp = ASSP(in_channels=256, out_channels=256) #20

      self.Doub0 = DoubleConv(512,256)
      self.Doub1 = DoubleConv(384, 128)
      self.Doub2 = DoubleConv(192, 128)
      self.Doub3 = DoubleConv(128, 64)
      self.Doub4 = nn.Conv2d(64,1,stride=1,kernel_size=1,bias=False)
      self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
      self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
      self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
      self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

      self.conv = nn.Conv2d(256,1,stride=1,kernel_size=1,bias=False)


    def forward(self, x):
        _, _, h, w = x.shape
        x = torch.cat((self.conv1(x),self.conv2(x),self.conv3(x),self.conv4(x)),dim=1)
        x1 = self.relu(self.resnet.bn1(x))
        x2 = self.resnet.maxpool(x1)
        x2 = self.resnet.layer1(x2)
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3(x3)
        a5 = self.assp(x4)

        out = self.conv(a5)
        out = F.interpolate(out, size=(h, w), mode='bilinear')


        if x4.shape != a5.shape:
            a5 = TF.resize(a5, size=x4.shape[2:])
        up0 = torch.cat((x4,a5),1)
        up0 = self.Doub0(up0)
        up1 = self.up1(up0)

        if up1.shape != x3.shape:
            up1 = TF.resize(up1, size=x3.shape[2:])
        up1 = torch.cat((up1,x3),1)
        up1 = self.Doub1(up1)
        up2 = self.up2(up1)

        if up2.shape != x2.shape:
            up2 = TF.resize(up2, size=x2.shape[2:])
        up2 = torch.cat((up2,x2),1)
        up2 = self.Doub2(up2)
        up3 = self.up3(up2)

        if up3.shape != x2.shape:
            up3 = TF.resize(up3, size=x2.shape[2:])
        up3 = torch.cat((up3,x2),1)
        up3 = self.Doub3(up3)
        up4 = self.up4(up3)

        up4 = self.Doub4(up4)
        up4 = TF.resize(up4, size=x.shape[2:])

        return up4, out

if __name__ == '__main__':
    x = torch.randn((5, 3, 858, 471))
    model = ResNet()
    model.eval()
    preds = model(x)
    print(preds[0].shape, preds[1].shape)

