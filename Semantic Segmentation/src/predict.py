import mindspore
import numpy as np
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter
from model_unet import UNet
from PIL import Image
import cv2

# 加载模型和参数
net = UNet(3, 1)
mindspore.load_checkpoint("../data/best_UNet.ckpt", net=net)

# 读取待预测图片
# img_path = '../data/ISIC2018_Task1/xxxL/val_origin/images/ISIC_0012643.jpg'
img_path = '../data/ISIC2018_Task1/huaweicloud/train/images/ISIC_0000001.jpg'
img = Image.open(img_path).convert('RGB')

# 处理图片
def val_transforms(img_size):
    return [
        vision.Resize(img_size, interpolation=Inter.NEAREST),
        vision.Rescale(1./255., 0.0),  # 归一化
        vision.HWC2CHW()
    ]
f_list = val_transforms((224, 224))

for f in f_list:
    img = f(img)

input = mindspore.Tensor(np.expand_dims(img, axis=0), mindspore.float32)

# 预测
pred = net(input)
pred[pred > 0.5] = float(1)
pred[pred <= 0.5] = float(0)

preds = np.squeeze(pred, axis=0)
img = np.transpose(preds,(1, 2, 0))

# 显示
cv2.imwrite('./predict.png', img.asnumpy()*255.)