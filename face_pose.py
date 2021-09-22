import torch.nn.functional as F
import torch.nn as nn
import sys

class backBone(nn.Module):
    def __init__(self):
        super(backBone, self).__init__()

        layer0 = Conv2dBatchReLU(1, 32, 3, 2)       #64
        layer1 = Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = Conv2dBatchReLU(64, 64, 3, 2)      #32
        layer3 = Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = Conv2dBatchReLU(128, 128, 3, 2)    #16

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def forward(self, x):

        x = self.layers(x)

        return x

class GlobalAvgPool2d(nn.Module):
    """ This layer averages each channel to a single number.
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        #print("gapSize: ", B, C, H, W, x.size())
        x = x.view(B, C, 1, 1)
        #print("gap2d", x.size())
        return x

class FullyConnectLayer(nn.Module):
    """ This layer averages each channel to a single number.
    """
    def __init__(self, in_channels, out_channels):
        super(FullyConnectLayer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )

    def forward(self, x):
        if len(x.size()) < 2:
            print("FullyConnectLayer input error!\n")
            sys.exit()
        flattenNum = 1
        for i in range(1,len(x.size())):
            flattenNum *= x.size(i)

        x = x.view(-1, flattenNum)
        x = self.layers(x)
        return x

class PoseBone(nn.Module):
    def __init__(self):
        super(PoseBone, self).__init__()

        self.features = nn.Sequential(
            Conv2dBatchReLU(128, 128, 3, 1),  # 16
            Conv2dBatchReLU(128, 256, 3, 2),  # 8
            Conv2dBatchReLU(256, 256, 3, 1),  # 8
            Conv2dBatchReLU(256, 512, 3, 2),  # 4
            Conv2dBatchReLU(512, 1024, 1, 1),  # 4
            GlobalAvgPool2d(),  # 1
        )

        self.dropout = nn.Dropout(0.5)

        self.yaw = FullyConnectLayer(1024, 120)
        self.pitch = FullyConnectLayer(1024, 66)
        self.roll = FullyConnectLayer(1024, 66)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        yaw = self.yaw(x)
        pitch = self.pitch(x)
        roll = self.roll(x)

        return yaw, pitch, roll


class pose(nn.Module):

    def __init__(self):
        super().__init__()

        self.baseBone = backBone()

        self.poseBone = PoseBone()

    def forward(self, x):
        x = self.baseBone(x)
        yaw, pitch, roll = self.poseBone(x)

        return [yaw, pitch, roll]


class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, isPadding=True, isBias=False):
        super(Conv2dBatchReLU, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.isBias = isBias
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)

        # Layer
        if isPadding == True:

            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                          bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
        else:

            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 0, bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):

        x = self.layers(x)

        return x