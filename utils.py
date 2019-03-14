import os
import math
import torch
from torchvision import datasets, transforms

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None, method='cosine'):
    """

    :param optimizer:
    :param epoch:
    :param args:
    :param batch:
    :param nBatch:
    :param method:
    :return:
    """
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def load_checkpoint(args):
    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res