import numpy as np
from mindspore import nn
from mindspore import Tensor
from mindspore._checkparam import Validator as validator

class MyMetric(nn.Metric):
    """
    评估指标:https://www.mindspore.cn/tutorials/zh-CN/r1.10/advanced/model/metric.html
    !!! 自定义Metrics函数需要继承nn.Metric父类，并重新实现父类中的clear方法、update方法和eval方法。(抽象方法)
    """
    def __init__(self, metrics= ["acc", "iou", "dice", "sens", "spec"], smooth=1e-5):
        super(MyMetric, self).__init__()
        self.metrics = metrics
        self.smooth = validator.check_positive_float(smooth, "smooth")
        self.metrics_list = [0. for i in range(len(self.metrics))]  # 全0
        self._samples_num = 0
        self.clear()  # 在每个训练或验证迭代结束后重置指标结果，以便于下一次迭代的计算。

    def Acc_metrics(self,y_pred, y):
        # 判断预测结果与真实结果完全相同有几个；TP表示真正例
        tp = np.sum(y_pred.flatten() == y.flatten(), dtype=y_pred.dtype)
        total = len(y_pred.flatten())
        single_acc = float(tp) / float(total)
        return single_acc

    def IoU_metrics(self,y_pred, y):
        """
        IoU = TP / (TP + FP + FN)  交集 / 并集
        """
        intersection = np.sum(y_pred.flatten() * y.flatten())  # 交集，切记像素点只有0 1
        # print(intersection)  全是0 ？？？？？
        unionset = np.sum(y_pred.flatten() + y.flatten()) - intersection
        single_iou = float(intersection) / float(unionset + self.smooth)
        return single_iou
    
    def Dice_metrics(self,y_pred, y):
        """
        Dice = (2 * TP) / (2 * TP + FP + FN), Dice系数，与交并比类似，便于求值。
        """
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten()) + np.sum(y.flatten())
        single_dice = 2*float(intersection) / float(unionset + self.smooth)
        return single_dice

    def Sens_metrics(self,y_pred, y):
        tp = np.sum(y_pred.flatten() * y.flatten())
        actual_positives = np.sum(y.flatten())
        single_sens = float(tp) / float(actual_positives + self.smooth)
        return single_sens
    
    def Spec_metrics(self, y_pred, y):
        true_neg = np.sum((1 - y.flatten()) * (1 - y_pred.flatten()))
        total_neg = np.sum((1 - y.flatten()))
        single_spec = float(true_neg) / float(total_neg + self.smooth)
        return single_spec
    
    def clear(self):
        """
        首先将self.metrics_list中的所有指标值设置为0，
        以清除模型在之前评估过程中的累计结果。
        然后，将样本数（self._samples_num）设置为0，以便于重新开始计算模型的指标。
        """
        self.metrics_list = [0. for i in range(len(self.metrics))]
        self._samples_num = 0

    def update(self, *inputs):
        """
        数据输入，执行评估指标，并将结果记录到self.metrics_list中
        """
        if len(inputs) != 2:
            raise ValueError("For 'update', it needs 2 inputs (predicted value, true value), ""but got {}.".format(len(inputs)))

        
        y_pred = Tensor(inputs[0]).asnumpy()  #modelarts,cpu
        # y_pred = np.array(Tensor(inputs[0]))  #cpu
        
        # 像素点 概率值转换为0 or 1
        y_pred[y_pred > 0.5] = float(1)
        y_pred[y_pred <= 0.5] = float(0)
        
        y = Tensor(inputs[1]).asnumpy() 
        self._samples_num += y.shape[0]

        if y_pred.shape != y.shape:
            raise ValueError(f"For 'update', predicted value (input[0]) and true value (input[1]) "
                             f"should have same shape, but got predicted value shape: {y_pred.shape}, "
                             f"true value shape: {y.shape}.")

        for i in range(y.shape[0]):
            if "acc" in self.metrics:
                single_acc = self.Acc_metrics(y_pred[i], y[i])
                self.metrics_list[0] += single_acc
            if "iou" in self.metrics:
                single_iou = self.IoU_metrics(y_pred[i], y[i])
                self.metrics_list[1] += single_iou
            if "dice" in self.metrics:
                single_dice = self.Dice_metrics(y_pred[i], y[i])
                self.metrics_list[2] += single_dice
            if "sens" in self.metrics:
                single_sens = self.Sens_metrics(y_pred[i], y[i])
                self.metrics_list[3] += single_sens
            if "spec" in self.metrics:
                single_spec = self.Spec_metrics(y_pred[i], y[i])
                self.metrics_list[4] += single_spec

    def eval(self):
        # 输出评估指标
        if self._samples_num == 0:
            raise RuntimeError("The 'metrics' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, or has "
                               "called update method before calling eval method.")
        for i in range(len(self.metrics_list)):
            self.metrics_list[i] = self.metrics_list[i] / float(self._samples_num)

        return self.metrics_list
    


# # test
# x = Tensor(np.array([[[[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.8]]]]))
# y = Tensor(np.array([[[[0, 1, 1], [1, 0, 0], [0, 1, 1]]]]))
# metric = MyMetric(["acc", "iou", "dice", "sens", "spec"], smooth=1e-5)
# metric.clear()
# metric.update(x, y)
# res = metric.eval()
# print( '丨acc: %.4f丨丨iou: %.4f丨丨dice: %.4f丨丨sens: %.4f丨丨spec: %.4f丨' % (res[0], res[1], res[2], res[3],res[4]), flush=True)