import os
import math
import torch
import operator
from functools import reduce
from torch.autograd import Variable
from torch.utils.data import Subset
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


def read_train_data(datadir, data):
    val_size = 5000
    test_size = 1000
    data_dir = os.path.join(datadir, data)
    if data == 'cifar10':
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
        indices = torch.randperm(len(train_set))
        train_indices = indices[:len(indices) - val_size]
        valid_indices = indices[len(indices) - val_size:]
        train_set = Subset(train_set, train_indices)
        val_set = Subset(val_set, valid_indices)
    elif data == 'cifar100':
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


def read_test_data(datadir, data, mode):
    data_dir = os.path.join(datadir, data)
    if data == 'cifar10':
        image_size = 32
        mean=[0.4914, 0.4824, 0.4467]
        std=[0.2471, 0.2435, 0.2616]
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std),
                                            ])
        test_set = datasets.CIFAR10(data_dir, train=False, transform=test_transforms, download=False)
    elif data == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std),
                                              ])
        test_set = datasets.CIFAR10(data_dir, train=False, transform=test_transforms, download=False)

    elif data == 'imagenet':
        pass

    else:
        raise NotImplementedError
    return test_set



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



def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_learned_conv
    elif type_name in ['LearnedGroupConv']:
        measure_layer(layer.relu, x)
        measure_layer(layer.norm, x)
        conv = layer.conv
        out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
                    conv.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
                    conv.stride[1] + 1)
        delta_ops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
                conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
        delta_params = get_layer_param(conv) / layer.condense_factor

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_num_gen(gen):
    return sum(1 for x in gen)


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])