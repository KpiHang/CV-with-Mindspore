import mindspore
import ml_collections
from model_unet import UNet
from data_loader import create_dataset
import mindspore.context as context
from mindspore import ops
from mindspore import nn
from metric import MyMetric


def train_model(net, train_loader, criterion, optimizer, num_epochs, device):
    # https://www.mindspore.cn/docs/zh-CN/r1.10/api_python/mindspore/mindspore.set_context.html?highlight=context#mindspore.set_context
    context.set_context(device_target=device)


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
    
    # 模型训练
    def train(epoch):
        """模型训练"""
        losses = []
        net.set_train(True)
        for i, (images, labels) in enumerate(train_loader):
            loss = train_step(images, labels)
            # if i % 100 == 0 or i == step_size_train - 1:
            print('Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.10f]' %
                    (epoch + 1, num_epochs, i + 1, train_loader.dataset.get_dataset_size(), loss))
            losses.append(loss)
        return sum(losses) / len(losses)

    # 验证集评估
    val_dataset = create_dataset(img_size=cfg.img_size, batch_size=cfg.val_batch_size, train_or_val='val', shuffle=True)
    val_size = val_dataset.get_dataset_size()
    def val(metrics=None):
        net.set_train(False)

        val_loader = val_dataset.create_tuple_iterator(num_epochs=cfg.val_epochs)

        val_loss = 0
        val_preds = []
        val_label = [] 
        for images, labels in val_loader:
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
        curr_loss = train(epoch)
        print("-" * 50)
        print("Epoch: [%3d/%3d], Average Train Loss: [%5.10f]" % (
            epoch+1, num_epochs, curr_loss
        ))
        metrics_name = ["acc", "iou", "dice", "sens", "spec"]
        iou_score, spec_score = val(metrics_name)
        if epoch > 2 and spec_score > 0.2:
            if iou_score > best_iou:
                best_iou = iou_score
                mindspore.save_checkpoint(net, ckpt_path)
            else:
                print('IoU did not improve from %0.4f' % best_iou)
        print("-" * 50)

def get_config():
    """configuration """
    config = ml_collections.ConfigDict()
    # net
    config.in_channel = 3
    config.n_classes = 1

    # dataset
    config.train_epochs = 3
    # config.train_data_path = "src/datasets/ISBI/train/"
    # config.val_data_path = "src/datasets/ISBI/val/"
    config.img_size = (224, 224)
    config.train_batch_size = 8

    config.val_epochs = 1
    config.val_batch_size = 4
    
    # train
    config.lr = 0.3
    config.device = "CPU"
    return config
    
if __name__ == "__main__":
    cfg = get_config()
    net = UNet(in_ch=cfg.in_channel, out_ch=cfg.n_classes)
    train_dataset = create_dataset(img_size=cfg.img_size, batch_size=cfg.train_batch_size, train_or_val='train', shuffle=True)
    train_loader = train_dataset.create_tuple_iterator(num_epochs=cfg.train_epochs)  # epochs 要遍历整个数据集几遍。
    # 优化器 https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Adam.html#mindspore.nn.Adam
    optimizer = nn.SGD(net.trainable_params(), learning_rate=cfg.lr)
    # 定义Loss
    criterion = nn.BCEWithLogitsLoss()

    train_model(net,
          train_loader,
          criterion,
          optimizer,
          cfg.train_epochs,
          device=cfg.device,
          )