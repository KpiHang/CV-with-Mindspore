import os
import glob
import mindspore
import numpy as np
from mindspore.dataset import vision, transforms
from mindspore.dataset.vision import Inter
from mindspore.dataset import GeneratorDataset, Dataset
import cv2
import random

class ISIC2018Dataset:
    def __init__(self, data_dir):
        self.image_files = glob.glob(os.path.join(data_dir, 'images') + '/*.jpg')
        self.mask_files = glob.glob(os.path.join(data_dir, 'masks') + '/*.png')

    def __getitem__(self, index):
        image = cv2.imread(self.image_files[index])
        label = cv2.imread(self.mask_files[index], cv2.IMREAD_GRAYSCALE)  # 单通道 2值类型
        label = label.reshape((label.shape[0], label.shape[1], 1))  # 如果不加这一步就变成2维的，1被省略了
        return image, label
    
    def __len__(self):
        return len(self.image_files)


def train_transforms(img_size):
    return [
    vision.Resize(img_size, interpolation=Inter.NEAREST),  # 变换像素，插值策略
    vision.Rescale(1./255., 0.0),  # 归一化
    vision.RandomHorizontalFlip(prob=0.5),  # 根据概率水平翻转
    vision.RandomVerticalFlip(prob=0.5),    # 根据概率垂直翻转
    vision.HWC2CHW()    # 将HWC转换为CHW 高宽通道数 转换为 通道数 高宽
    ]

def val_transforms(img_size):  # 训练集 由于数据增广的需要，验证集不需要
    return [
    vision.Resize(img_size, interpolation=Inter.NEAREST),
    vision.Rescale(1./255., 0.0),  # 归一化
    vision.HWC2CHW()
    ]

def create_dataset(img_size=(224, 224), batch_size=8, train_or_val='train', shuffle=True, num_workers=1):
    """
    创建一个数据集，使用ISIC2018Dataset类并对图像和标签应用转换。

    :param img_size: 一个元组，指定转换后的图像尺寸。
    :type img_size: tuple, 可选

    :param train_or_val: 一个字符串，指示是否创建训练或验证数据集。
    :type train_or_val: str, 可选

    :param shuffle: 一个布尔值，指示是否应对数据集进行洗牌。
    :type shuffle: bool, 可选

    :returns: 经过转换的图像和标签的数据集对象。
    :rtype: Dataset
    """
    data = ISIC2018Dataset(data_dir='../data/ISIC2018_Task1/' + train_or_val)
    dataset = GeneratorDataset(source=data, column_names=["image", "label"], shuffle=shuffle)
    
    if train_or_val == 'train':
        transform_img = train_transforms(img_size)
    else:
        transform_img = val_transforms(img_size)

    dataset = dataset.map(input_columns='image', num_parallel_workers=1, operations=transform_img)
    dataset = dataset.map(input_columns="label", num_parallel_workers=1, operations=transform_img)
    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_workers=num_workers)
    print(train_or_val + ' set shape:', dataset.get_dataset_size())
    return dataset



# # dataset 测试
# dataset = create_dataset(train_or_val='val')
# image, label = next(dataset.create_tuple_iterator())
# print(image, image.shape, image.dtype)
# print(label.shape, sum(label))

# dataset = create_dataset(train_or_val='val')
# loder = dataset.create_tuple_iterator()
# print(dataset.get_dataset_size(), loder.dataset.get_dataset_size())