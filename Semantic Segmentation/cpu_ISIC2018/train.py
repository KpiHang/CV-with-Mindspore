import mindspore
from model_unet import UNet
from data_loader import create_dataset
import mindspore.context as context
from mindspore import ops
from mindspore import nn
from metric import MyMetric






def train_unet(net, device='cpu', num_epochs=1, batch_size=16, lr=0.00001):
    # https://www.mindspore.cn/docs/zh-CN/r1.10/api_python/mindspore/mindspore.set_context.html?highlight=context#mindspore.set_context
    context.set_context(device_target=device)

    # 加载数据集
    train_dataset = create_dataset(img_size=(224, 224), batch_size=batch_size, train_or_val='train', shuffle=True)
    step_size_train = train_dataset.get_dataset_size()
    train_loader = train_dataset.create_tuple_iterator(num_epochs=num_epochs)  # epochs 要遍历整个数据集几遍。

    val_dataset = create_dataset(img_size=(224, 224), batch_size=8, train_or_val='val', shuffle=True)
    dataloader = val_dataset.create_tuple_iterator(num_epochs=1)
    val_size = val_dataset.get_dataset_size()

    # 定义优化器
    # https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Adam.html#mindspore.nn.Adam
    optimizer = nn.SGD(net.trainable_params(), learning_rate=lr)
    # optimizer = nn.RMSProp(net.parameters(),learning_rate=lr,momentum=0.9,weight_decay=1e-8)
    
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()

    # 定义前向传播
    def forward_fn(inputs, targets):
        logits = net(inputs)
        loss = criterion(logits, targets)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    # 参数更新
    def train_step(inputs, targets):
        loss, grads = grad_fn(inputs, targets)
        optimizer(grads)
        return loss

    def train(data_loader, epoch):
        """模型训练"""
        losses = []
        net.set_train(True)

        for i, (images, labels) in enumerate(data_loader):
            loss = train_step(images, labels)
            # if i % 100 == 0 or i == step_size_train - 1:
            print('Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.10f]' %
                    (epoch + 1, num_epochs, i + 1, step_size_train, loss))
            losses.append(loss)
        return sum(losses) / len(losses)
    

    def val(dataloader, metrics=None):
        net.set_train(False)

        val_loss = 0
        val_preds = []
        val_label = [] 
        for images, labels in dataloader:
            y_pred = net(images)
            val_loss += criterion(y_pred, labels).asnumpy()
            val_preds.append(y_pred.asnumpy())
            val_label.append(labels.asnumpy())
        
        val_loss /= val_size
        metric = MyMetric(metrics, smooth=1e-5)
        metric.clear
        metric.update(val_preds, val_label)
        res = metric.eval()
        
        print(f'Val loss:{val_loss:>4f}','丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))
        
        iou_score = res[1]
        spec_score = res[4]
        return iou_score, spec_score

    # 开始训练
    print("Start Training Loop ...")
    best_iou = 0
    ckpt_path = 'best_UNet.ckpt'
    for epoch in range(num_epochs):
        curr_loss = train(train_loader, epoch)
        print("-" * 50)
        print("Epoch: [%3d/%3d], Average Train Loss: [%5.10f]" % (
            epoch+1, num_epochs, curr_loss
        ))
        metrics_name = ["acc", "iou", "dice", "sens", "spec"]
        iou_score, spec_score = val(dataloader, metrics_name)
        if epoch > 2 and spec_score > 0.2:
            if iou_score > best_iou:
                best_iou = iou_score
                mindspore.save_checkpoint(net, ckpt_path)
            else:
                print('IoU did not improve from %0.4f' % (best_iou) % best_iou)
        print("-" * 50)




if __name__ == "__main__":
    net = UNet(in_ch=3, out_ch=1)
    device = 'cpu'
    train_unet(net, 
          device,
          epochs=5,
          batch_size=8,
          lr=0.01)