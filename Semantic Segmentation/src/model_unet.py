import mindspore
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor

class DoubleConv(nn.Cell):
    """
    双层卷积模块，UP 和 Down 都用到了，两个3x3卷积。
    """
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch), nn.ReLU()
        )

    def construct(self, x):
        x = self.conv(x)
        return x


class Down(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.SequentialCell(
            [nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channels, out_channels)]
        )
    
    def construct(self, x):
        x = self.down(x)
        return x

class Up(nn.Cell):
    """
    上采样模块，使用双线性插值实现上采样。
    """
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ResizeBilinear()
        self.conv = DoubleConv(in_ch, out_ch)

    def construct(self, x1, x2):
        x1 = self.up(x1, scale_factor=2)
        # x = np.concatenate((x2, x1), axis=1)  # [N,3,256,256] 1 表示通道数所在维数，Unet U的来源。
        x = ops.concat((x2, x1), 1)
        x = self.conv(x)
        return x


class UNet(nn.Cell):
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()

        # Down
        self.inchannel = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # UP
        self.up1 = Up(1024+512, 512)
        self.up2 = Up(512+256, 256)
        self.up3 = Up(256+128, 128)
        self.up4 = Up(128+64, 64)
        self.final = nn.Conv2d(64, out_ch, kernel_size=1)  # (64 1 1) 
        
        # output
        self.sigmoid = ops.Sigmoid()  # out每个像素点是一个概率值 到0-1之间，类似2分类

    def construct(self, x):
        x1 = self.inchannel(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final(x)
        output = self.sigmoid(x)
        return output


# # 模型测试
# if __name__ == '__main__':
#     img_size = 256
#     x = Tensor(np.zeros([10,3,img_size,img_size]), mindspore.float32)
#     model = UNet()
#     output = model(x)   # output.shape (B,3,512,512)
#     print(output)
#     print(output.shape)