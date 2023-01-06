'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import os
import argparse
import utils
from fer import FERData
from torch.autograd import Variable
import datetime

begin_time = datetime.datetime.now()

parser = argparse.ArgumentParser(description='PyTorch VGG Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='BAUM', help='CNN architecture')
parser.add_argument('--size', default=224, type=int, help='img size')
parser.add_argument('--bs', default=64, type=int, help='learning rate')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
model_name = "vgg"
num_classes = 6

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
total_epoch = 50 #250

path = os.path.join(opt.dataset + '_' + opt.model)
if not(os.path.exists(path)):
    os.mkdir(path)
feature_extract = True

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



# Model
# if opt.model == 'VGG19':
#     net = VGG('VGG19')
# elif opt.model  == 'Resnet18':
#     net = ResNet18()
# net = VGG(opt.model)
# if use_cuda:
#     net.cuda()
# net = nn.DataParallel(net, device_ids = device_ids)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = opt.size

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = opt.size

    elif model_name == "vgg":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = opt.size

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = opt.size

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = opt.size

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = opt.size

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

     
# optimizer = nn.DataParallel(optimizer, device_ids = device_ids)
# Training
def train(epoch):
    print('\nEpoch: %d' % (epoch))
    global Train_acc
    model_ft.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer_ft, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer_ft.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model_ft(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer_ft, 0.1)
        optimizer_ft.step()
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
    model_ft.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model_ft(inputs)
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
            'model_ft': model_ft.state_dict() if use_cuda else model_ft,
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
    model_ft.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model_ft(inputs)
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
            'model_ft': model_ft.state_dict() if use_cuda else model_ft,
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
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft.cuda()
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    optimizer_ft = optim.SGD(params_to_update, lr=opt.lr, momentum=0.9)
    acc_total = []

    SAVE_DATE = str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().day) + '_'
    Data_USE = opt.dataset + '_img_6emo_' + opt.size + '_finetune_classifier'
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

        model_ft.load_state_dict(checkpoint['model_ft'])
        best_PublicTest_acc = checkpoint['best_PublicTest_acc']
        best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
        best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
        best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
        start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
    else:
        print('==> Building model..')

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
    result_txt = open('./result/' + opt.model + '/' + SAVE_NAME[:-9] + '_finetune_fc_result.txt', 'a')
    result_txt.write('\nbest_PublicTest_acc: %0.3f' % best_PublicTest_acc)
    result_txt.write('\nbest_PublicTest_acc_epoch: %d' % best_PublicTest_acc_epoch)
    result_txt.write('\nbest_PrivateTest_acc: %0.3f' % best_PrivateTest_acc)
    result_txt.write('\nbest_PrivateTest_acc_epoch: %d' % best_PrivateTest_acc_epoch)
    result_txt.write('\nbest_Pc: ' + str(best_pc))
    result_txt.write('\n-----------------------------------------------')

