# -*- codeing = utf-8 -*-
# Time : 2023/6/14 13:00
# @Auther : zhouchao
# @File: DeepLearning.py
# @Software:PyCharm
import torch
import pickle
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
from resnet import resnet18
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from utils import read_data
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


log_dir = "./runs/resnet/"
save_best_path = "./models/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

print(device)
Batch_Size = 512
epochs = 10000


raw_path = 'data/hebing.raw'
rgb_path = 'data/hebing.png'
bands = [33, 34, 35, 36, 37, 38, 39]
data_x, data_y = read_data(raw_path, rgb_path, shape=(692, 272, 768), setect_bands=bands, blk_size=5, cut_shape=(690, 765))
data_x_shape = data_x.shape
data_x = data_x.reshape(-1, data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
rus = RandomUnderSampler(random_state=0)
data_x, data_y = rus.fit_resample(data_x, data_y)
data_x = data_x.reshape(-1, data_x_shape[1], data_x_shape[2], data_x_shape[3]).transpose(0, 3, 1, 2)
# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
train_x = np.asarray(train_x, dtype=np.float32)
train_y = np.asarray(train_y, dtype=np.int)
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
train_y = train_y.to(torch.int64)
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=Batch_Size, num_workers=0)

test_x = np.asarray(test_x, dtype=np.float32)
test_y = np.asarray(test_y, dtype=np.int)
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y)
test_y = test_y.to(torch.int64)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=Batch_Size, num_workers=0)

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print('训练集的大小：' + str(train_data_size))
print('测试集的大小：' + str(test_data_size))

net = resnet18(num_classes=6)
net = net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)
history = []

best_acc = 0.0
best_epoch = 0
for epoch in range(epochs):
    epoch_start = time.time()
    print(f'Epoch:{epoch + 1}/{epochs}')
    net.train()
    train_loss = 0.0
    train_acc = 0.0
    test_loss = 0.0
    test_acc = 0.0
    for i, (train_data, train_label) in enumerate(train_loader):
        inputs = train_data.to(device)
        labels = train_label.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc += acc.item() * inputs.size(0)

    net.eval()
    acc = 0.0
    t1 = time.time()
    with torch.no_grad():
        for j, (test_data, test_label) in enumerate(test_loader):
            inputs = test_data.to(device)
            labels = test_label.to(device)
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            test_acc += acc.item() * inputs.size(0)
    print(f"预测时间{time.time() - t1}")
    avg_train_loss = train_loss / train_data_size
    avg_train_acc = train_acc / train_data_size

    avg_test_loss = test_loss / test_data_size
    avg_test_acc = test_acc / test_data_size
    # 将每一轮的损失值和准确率记录下来
    history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])
    if best_acc < avg_test_acc:
        best_acc, best_epoch = avg_test_acc, epoch
        path = os.path.join(save_best_path, f'best_5x5_resnet.pth')
        torch.save(net.state_dict(), path)
    epoch_end = time.time()
    writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar("Train_loss", history[-1][0], epoch)
    writer.add_scalar("Train_accuracy", history[-1][2], epoch)
    writer.add_scalar("Val_loss", history[-1][1], epoch)
    writer.add_scalar("Val_accuracy", history[-1][3], epoch)
    writer.add_scalar("Best_acc", best_acc, epoch)
    writer.add_scalar("best_epoch", best_epoch, epoch)
    # 打印每一轮的损失值和准确率，效果最佳的验证集准确率
    print(
        "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch, avg_train_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100, epoch_end - epoch_start
        ))
    print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))