#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import datetime
import json
import os
import subprocess

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--val-size', default=0, type=int,
                    help='hold out this many training images (class-balanced) '
                         'for validation; the test set is then only evaluated '
                         'after the final epoch (default: 0, no split)')
parser.add_argument('--split-seed', default=7357, type=int,
                    help='seed for the train/validation split, kept separate '
                         'from --seed so every run gets the same split')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy (validation accuracy with --val-size)
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
                            transform=transform_train)

testset = datasets.CIFAR10(root='~/data', train=False, download=True,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=8)


def split_indices(targets, val_size, seed):
    '''Class-balanced split of the training set into train and val indices.'''
    num_classes = len(set(targets))
    assert val_size % num_classes == 0, \
        '--val-size must be divisible by the number of classes'
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    train_idx, val_idx = [], []
    for c in range(num_classes):
        idx = rng.permutation(np.where(targets == c)[0])
        val_idx.extend(idx[:val_size // num_classes].tolist())
        train_idx.extend(idx[val_size // num_classes:].tolist())
    return sorted(train_idx), sorted(val_idx)


if args.val_size > 0:
    train_idx, val_idx = split_indices(trainset.targets, args.val_size,
                                       args.split_seed)
    # Save the split so it can be checked, and refuse to silently use a
    # different one if the splitting code ever changes.
    if not os.path.isdir('results'):
        os.mkdir('results')
    splitname = 'results/split_val%d_seed%d.json' % (args.val_size,
                                                     args.split_seed)
    if os.path.exists(splitname):
        with open(splitname) as splitfile:
            saved = json.load(splitfile)
        assert saved == {'train': train_idx, 'val': val_idx}, \
            'Error: split differs from ' + splitname
    else:
        with open(splitname, 'w') as splitfile:
            json.dump({'train': train_idx, 'val': val_idx}, splitfile)

    # Validation and the clean training-set evaluation use the test
    # transform: original, unaugmented, unmixed images.
    cleanset = datasets.CIFAR10(root='~/data', train=True, download=True,
                                transform=transform_test)
    valloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cleanset, val_idx), batch_size=100,
        shuffle=False, num_workers=8)
    cleantrainloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cleanset, train_idx), batch_size=100,
        shuffle=False, num_workers=8)
    trainset = torch.utils.data.Subset(trainset, train_idx)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # The checkpoint pickles the whole model, so full unpickling is required.
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed), weights_only=False)
    net = checkpoint['net']
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')


def git_output(*cmd):
    try:
        return subprocess.check_output(('git',) + cmd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# Record the hyperparameters and environment next to the log. Resumed or
# repeated runs append a session, matching how rows are appended to the CSV.
metaname = os.path.splitext(logname)[0] + '.json'
meta = {'sessions': []}
if os.path.exists(metaname):
    with open(metaname) as metafile:
        meta = json.load(metafile)
git_status = git_output('status', '--porcelain', '--untracked-files=no')
meta['sessions'].append({
    'args': vars(args),
    'started': datetime.datetime.now().isoformat(timespec='seconds'),
    'start_epoch': start_epoch,
    'git_commit': git_output('rev-parse', '--short', 'HEAD'),
    'git_dirty': bool(git_status) if git_status is not None else None,
    'torch': torch.__version__,
    'gpu': torch.cuda.get_device_name(0) if use_cuda else None,
})
with open(metaname, 'w') as metafile:
    json.dump(meta, metafile, indent=2)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item()
                    + (1 - lam) * predicted.eq(targets_b).sum().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
            100.*correct/total)


def evaluate(loader):
    '''Returns mean loss and accuracy of the model on unmixed images.'''
    net.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (eval_loss/(batch_idx+1), 100.*correct/total,
                            correct, total))
    return (eval_loss/(batch_idx+1), 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if args.val_size > 0:
    header = ['epoch', 'train loss', 'reg loss', 'train acc',
              'clean train loss', 'clean train acc', 'val loss', 'val acc']
else:
    header = ['epoch', 'train loss', 'reg loss', 'train acc',
              'test loss', 'test acc']
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(header)

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    row = [epoch, train_loss, reg_loss, train_acc]
    if args.val_size > 0:
        print('Clean train:')
        row += evaluate(cleantrainloader)
        print('Val:')
        loss, acc = evaluate(valloader)
    else:
        loss, acc = evaluate(testloader)
    row += [loss, acc]
    # Checkpoint selection uses validation accuracy when there is a split,
    # so the test set never influences training.
    if epoch == args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(row)

if args.val_size > 0 and start_epoch < args.epoch:
    # The held-out test set is evaluated once, on the final-epoch model.
    print('Test (final epoch):')
    test_loss, test_acc = evaluate(testloader)
    meta['sessions'][-1]['final_test'] = {'epoch': args.epoch - 1,
                                          'loss': test_loss, 'acc': test_acc}
    with open(metaname, 'w') as metafile:
        json.dump(meta, metafile, indent=2)
    print('Final test accuracy: %.2f%% (error %.2f%%)'
          % (test_acc, 100. - test_acc))
