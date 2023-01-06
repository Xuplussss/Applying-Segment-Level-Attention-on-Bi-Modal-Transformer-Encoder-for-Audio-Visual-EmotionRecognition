'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import numpy as np
import os
import argparse
import utils
from fer import FERData
from torch.autograd import Variable
from models import *
import datetime
import matplotlib.pyplot as plt

begin_time = datetime.datetime.now()

parser = argparse.ArgumentParser(description='PyTorch VGG Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='BAUM', help='CNN architecture')
parser.add_argument('--bs', default=32, type=int, help='learning rate')
parser.add_argument('--size', default=48, type=int, help='img size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_pc = 0

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 100 #250

path = os.path.join(opt.dataset + '_' + opt.model)
if not(os.path.exists(path)):
    os.mkdir(path)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])



# optimizer = nn.DataParallel(optimizer, device_ids = device_ids)
# Training
def train(epoch):
    print('\n Epoch: %d' % (epoch))
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Train_acc = 100.*correct/total

def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    global best_pc
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PublicTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    pc = correct/total
    PublicTest_acc = 100.*correct/total
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, SAVE_NAME))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch
    if pc > best_pc:
        best_pc = pc
        print('pc:',pc)

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total

    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, SAVE_NAME))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

if __name__ == '__main__':

    net = VGG(opt.model)
    if use_cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    acc_total = []
    SAVE_DATE = str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().day) + '_'
    Data_USE = opt.dataset + '_img_6emo_' + opt.size + '_scratch'
    SAVE_NAME = SAVE_DATE + Data_USE + '.t7'
    trainset = FERData(split = 'Training', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
    PublicTestset = FERData(split = 'PublicTest', transform=transform_test)
    PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
    PrivateTestset = FERData(split = 'PrivateTest', transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(path, SAVE_NAME))

        net.load_state_dict(checkpoint['net'])
        best_PublicTest_acc = checkpoint['best_PublicTest_acc']
        best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
        best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
        best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
        start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
    else:
        print('==> Building model..')

    device_ids = [3,7]
    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        PublicTest(epoch)
        PrivateTest(epoch)
    end_time = datetime.datetime.now()
    acc_total.append(best_PublicTest_acc)
    if not(os.path.exists('./result/' + opt.model + '/')):
        os.mkdir('./result/' + opt.model + '/')
    print('training time: ',end_time - begin_time)
    print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
    print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
    print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
    print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
    result_txt = open('./result/' + opt.model + '/' + SAVE_NAME[:-9] + '_scratch_result.txt', 'a')
    result_txt.write('\nbest_PublicTest_acc: %0.3f' % best_PublicTest_acc)
    result_txt.write('\nbest_PublicTest_acc_epoch: %d' % best_PublicTest_acc_epoch)
    result_txt.write('\nbest_PrivateTest_acc: %0.3f' % best_PrivateTest_acc)
    result_txt.write('\nbest_PrivateTest_acc_epoch: %d' % best_PrivateTest_acc_epoch)
    result_txt.write('\nbest_Pc: ' + str(best_pc))
    result_txt.write('\n-----------------------------------------------')

