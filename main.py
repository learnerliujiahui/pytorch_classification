import os
import sys
import math
sys.path.append(r"/home/liujiahui/PycharmProjects/efficient_densenet_pytorch/models")
import argparse
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import models
from models import densenet, condensenet
from utils import AverageMeter, read_train_data, read_test_data, adjust_learning_rate, \
    load_checkpoint, accuracy, measure_model

# version pytorch 1.0

'''
set the data dir and saving dir 
'''
parser = argparse.ArgumentParser(description='Model Parameters')
parser.add_argument('--mode', default='train', choices=['train', 'test'])  # train/test
parser.add_argument('--featureVisualize', default=True)
parser.add_argument('--data', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--data_dir', default='/home/liujiahui/data_zoo/')  # + cifar10/cifar100/...
parser.add_argument('--save_dir', default='/home/liujiahui/PycharmProjects/efficient_densenet_pytorch/0315/model')
parser.add_argument('--log_dir', default='/home/liujiahui/PycharmProjects/efficient_densenet_pytorch/0315/log')
parser.add_argument('--para_dir', default='/home/liujiahui/PycharmProjects/efficient_densenet_pytorch/0315/para')
parser.add_argument('--gpu', default='0')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model', default='condensenet')  # DensrNet/ CondenseNet/ RsNet
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
parser.add_argument('--num_classes', default=10)
# for densenet
parser.add_argument('--depth', default=100)
parser.add_argument('--growth_rate', default=12)
# for condensenet
parser.add_argument('--stages', default='14-14-14')
parser.add_argument('--growth', default='8-16-32')
parser.add_argument('--reduction', default=0.5, type=float, metavar='R', help='transition reduction (default: 0.5)')
parser.add_argument('--dropout-rate', default=0, type=float, help='drop out (default: 0)')
parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO', help='group lasso loss weight (default: 0)')
parser.add_argument('--bottleneck', default=4)
parser.add_argument('--group-1x1', default=4, help='1x1 group convolution (default: 4)')
parser.add_argument('--group-3x3', default=4, help='3x3 group convolution (default: 4)')
parser.add_argument('--condense-factor', default=4, help='condense factor (default: 4)')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # set the GPU number
args.stages = list(map(int, args.stages.split('-')))
args.growth = list(map(int, args.growth.split('-')))

if args.condense_factor is None:
    args.condense_factor = args.group_1x1


def main():
    # set model
    model = getattr(models, args.model)(args)

    if args.data == 'cifar10':
        image_size = 32
        args.num_classes = 10
    elif args.data == 'cifar100':
        image_size = 32
        args.num_classes = 100
    elif args.data == 'imagenet':
        image_size = 224
        args.num_classes = 1000
    else:
        raise NotImplementedError

    n_flops, n_params = measure_model(model, image_size, image_size)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))

    if torch.cuda.device_count():
        model = torch.nn.DataParallel(model)  # for multi-GPU training
    if torch.cuda.is_available():
        model.cuda()

    print(model)

    if args.mode == 'train':
        # get the training loader and validation loader
        train_set, val_set = read_train_data(datadir=args.data_dir, data=args.data)

        # set the start epoch value
        if args.resume:
            start_epoch = None
        else:
            start_epoch = args.start_epoch

        train(startepoch=start_epoch, epochs=args.epochs, model=model, train_set=train_set,
              val_set=val_set, resume=args.resume)

    elif args.mode == 'test':
        test_set = read_test_data(datadir=args.data_dir, data=args.data, mode='test')
        test(model=model, test_set=test_set)
    else:
        raise NotImplementedError


def train(startepoch, epochs, model, train_set, val_set, resume):
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # set the loss function and optimizer(SGD)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # recover the parameters from checkpoint files
    if resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    elif not resume:
        start_epoch = startepoch
        Writer = SummaryWriter(log_dir=args.log_dir)

    else:
        raise NotImplementedError
    # set a variable for tensorboard
    training_total_loss = 0

    best_error = 100  # set the flag for beat parameters
    # starting training
    for epoch in range(start_epoch, epochs):
        # start training
        tr_error1, tr_error5, loss, lr, training_total_loss = train_eopch(train_loader, model, optimizer, criterion,
                                                                        epoch, epochs, training_total_loss)

        # record the training information into ckpt file --> TensorboardX
        Writer.add_scalar("training total loss", training_total_loss, epoch)
        Writer.add_scalar("training top-1 error", tr_error1, epoch)
        Writer.add_scalar("training top-5 error", tr_error5, epoch)
        Writer.add_scalar("learning rate", lr, epoch)

        print("the total loss of epoch {}, is: {}".format(epoch, training_total_loss))
        training_total_loss = 0

        # start validation
        val_error1, val_error5, loss, losses = val_epoch(val_loader, model)
        Writer.add_scalar("validation total loss", losses, epoch)
        Writer.add_scalar("validation top-1 error", val_error1, epoch)
        Writer.add_scalar("validation top-5 error", val_error5, epoch)

        is_best = val_error1 < best_error
        best_error = min(val_error1, best_error)
        if is_best:
            print("New best error:%.4f" % val_error1)
            torch.save(model.state_dict(), os.path.join(args.para_dir, 'model.dat'))


def train_eopch(train_loader, model, optimizer, criterion, epoch, epochs, total_loss):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    learned_module_list = []  # add all the LGC layers into the list

    # Switch to train mode
    model.train()
    # Find all learned convs to prepare for group lasso loss
    for m in model.modules():
        if m.__str__().startswith('LearnedGroupConv'):
            learned_module_list.append(m)

    running_lr = None

    beginTime = time.time()
    for i, (input, target) in enumerate(train_loader):
        progress = float(epoch * len(train_loader) + i) / (epochs * len(train_loader))
        args.progress = progress

        lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - beginTime)

        # input Tensor CPU --> GPU
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # Tensor --> Variable --> model
        inputVar = Variable(input, requires_grad=True)
        targetVar = Variable(target)

        # compute the result
        output, _ = model(inputVar, progress)
        loss = criterion(output, targetVar)

        # Add group lasso loss to the basic loss value
        if args.group_lasso_lambda > 0:
            lasso_loss = 0
            for m in learned_module_list:
                lasso_loss = lasso_loss + m.lasso_loss
            loss = loss + args.group_lasso_lambda * lasso_loss

        total_loss += loss.item()

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - beginTime)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'  # ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}\t'  # ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}\t'  # ({loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f}\t'  # ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f}\t'  # ({top5.avg:.3f})'
                  'lr {lr: .4f}'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))

    return top1.avg, top5.avg, losses.avg, lr, total_loss


def val_epoch(val_loader, model):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    model.eval()
    begin_time = time.time()
    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        criterion = nn.CrossEntropyLoss().cuda()

        # Compute output
        output, _ = model(input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - begin_time)
        end_time = time.time()

        if i % args.print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time,
                                                                  loss=losses, top1=top1, top5=top5))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, loss, losses


def test(model, test_set):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=args.works)

    # Switch to evaluate mode
    model.eval()

    begin_time = time.time()

    for i, (input, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        criterion = nn.CrossEntropyLoss().cuda()

        # Compute output
        output, featureMaps = model(input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - begin_time)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(test_loader), batch_time=batch_time,
                                                                  loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))


if __name__ =='__main__':
    """
    the args parameters must be set in advance
    """
    main()