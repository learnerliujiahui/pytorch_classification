# import fire
import os
import sys
import math
sys.path.append(r"/home/liujiahui/PycharmProjects/efficient_densenet_pytorch/models")
import argparse
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import models
from models import densenet, condensenet
from utils import AverageMeter


# version pytorch 1.0

'''
set the data dir and saving dir 
'''

parser = argparse.ArgumentParser(description='Model Parameters')
parser.add_argument('--mode', default='train', choices=['train', 'test'])  # train/test
parser.add_argument('--featureVisualize', default=True)
parser.add_argument('--data', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--data_dir', default='/home/liujiahui/data_zoo/')  # + cifar10/cifar100/...
parser.add_argument('--save_dir', default='/home/liujiahui/PycharmProjects/efficient_densenet_pytorch/log0310')
parser.add_argument('--para_dir', default='/home/liujiahui/PycharmProjects/efficient_densenet_pytorch/log0309')
parser.add_argument('--model', default='denseNet')  # DensrNet/ CondenseNet/ RsNet
parser.add_argument('--test_size', default=500)
parser.add_argument('--start_epoch', default=0)
parser.add_argument('--epochs', default=300)
parser.add_argument('--resume', default= False)
parser.add_argument('--efficient', default=True)
parser.add_argument('--batch_size',default=256)
# model hyper-parameters:
parser.add_argument('--lr', default=0.1, help='the intial learning rate value')
parser.add_argument('--lr_type', default='cosine', choices=['cosine', 'multistep'])

parser.add_argument('--weight_decay', default=1e-4)
parser.add_argument('--momentum', default=0.9)
# for densenet
parser.add_argument('--depth', default=100)
parser.add_argument('--growth_rate', default=12)
# for condensenet
parser.add_argument('--stage', default='14-14-14')
parser.add_argument('--growth', default='8-16-32')
parser.add_argument('--bottleneck', default=4)
parser.add_argument('--group-1x1', default=4, help='1x1 group convolution (default: 4)')
parser.add_argument('--group-3x3', default=4, help='3x3 group convolution (default: 4)')
parser.add_argument('--condense-factor', default=4, help='condense factor (default: 4)')

args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # set the GPU number
    block_config = list(map(int, args.stages.split('-')))
    block_growth = list(map(int, args.growth.split('-')))

    if args.condense_factor is None:
        args.condense_factor = args.group_1x1

    # set model
    model = getattr(models, args.model)(args)
    print(model)


    if args.mode == 'train':
        # get the training loader and validation loader
        train_set, val_set = read_train_data(datadir=args.data_dir, data=args.data, mode='train')

        # set the start epoch value
        if args.resume:
            start_epoch = None
        else:
            start_epoch = args.start_epoch

        train(start_epoch=start_epoch, epochs=args.epochs, model=model, train_set=train_set, val_set=val_set)

    elif args.mode =='test':
        test(model=model, )
    else:
        raise NotImplementedError



def train(start_epoch, epochs, model, train_set, val_set):
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        learned_module_list = []
        # Find all learned convs to prepare for group lasso loss
        for m in model.modules():
            if m.__str__().startswith('LearnedGroupConv'):
                learned_module_list.append(m)

        running_lr = None

        for i, (input, target) in enumerate(train_loader):
            if torch.cuda.is_avilable():
                input = input.cuda()
                target = target.cuda()

            input = Variable(input, requires_grad=True)
            target = Variable(target)
            loss =
            output, featureMaps = model(input, target)


def test():
    pass


def read_train_data(datadir, data, mode):
    data_dir = os.path.join(datadir, data)
    if data == 'cifar10':

        num_classes = 10
        image_size = 32
        mean=[0.4914, 0.4824, 0.4467]
        std=[0.2471, 0.2435, 0.2616]
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.Compose([
                                                                                    transforms.RandomCrop(image_size, padding=4),
                                                                                    transforms.RandomHorizontalFlip(),
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Normalize(mean=mean, std=std)]))
        val_set = datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Normalize(mean=mean, std=std)]))
    elif data == 'cifar100':
        num_classes = 100
        image_size = 32
        mean=[0.5071, 0.4867, 0.4408]
        std=[0.2675, 0.2565, 0.2761]
        train_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=transforms.Compose([
                                                                              transforms.RandomCrop(image_size, padding=4),
                                                                              transforms.RandomHorizontalFlip(),
                                                                              transforms.ToTensor(),
                                                                              transforms.Normalize(mean=mean, std=std)]))
        val_set = datasets.CIFAR100(data_dir, train=False, transform=transforms.Compose([
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean=mean, std=std)]))
    elif data == 'imagenet':
        num_classes = 1000
        image_size = 224
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        train_set = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomSizedCrop(image_size),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize(mean=mean, std=std)]))

        val_set = datasets.ImageFolder(valdir, transforms.Compose([transforms.Scale(256),
                                                                   transforms.CenterCrop(224),
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize(mean=mean, std=std)]))

    else:
        raise NotImplementedError

    return train_set, val_set




if __name__ =='__main__':
    main()