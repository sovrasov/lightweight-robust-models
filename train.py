import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from pytorchcv.model_provider import get_model, _models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.dataloaders import get_pytorch_train_loader, get_pytorch_val_loader
from tensorboardX import SummaryWriter

from utils.pgd import pgd_attack


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenetv2_w1',
                    choices=_models.keys(),
                    help='model architecture: ' +
                        ' | '.join(_models.keys()))
parser.add_argument('--arch-student', default='', type=str,
                    choices=_models.keys(),
                    help='model architecture: ' +
                        ' | '.join(_models.keys()))

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--adv-eps', type=float, default=0.01)
parser.add_argument('--euclidean', dest='euclidean', action='store_true')

parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

parser.add_argument('--oi-list', type=str, default='',
                    help='path to a custom list with OI images')
parser.add_argument('--oi-thresh', type=float, default=0.4)

parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')
parser.add_argument('--input-size', type=int, default=224, help='MobileNet model input resolution')
parser.add_argument('--weight', default='', type=str, metavar='WEIGHT',
                    help='path to pretrained weight (default: none)')


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    print("=> creating model '{}'".format(args.arch))
    models = [get_model(args.arch, in_size=(args.input_size, args.input_size), num_classes=1000, pretrained=False)]
    if len(args.arch_student):
        print("=> creating model '{}'".format(args.arch_student))
        models.append(get_model(args.arch_student, in_size=(args.input_size, args.input_size), num_classes=1000, pretrained=False))


    if not args.distributed:
        models = [torch.nn.DataParallel(model).cuda() for model in models]
    else:
        models = [torch.nn.parallel.DistributedDataParallel(model.cuda()) for model in models]

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizers = [torch.optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay) for model in models]

    # optionally resume from a checkpoint
    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            models[0].load_state_dict(checkpoint['state_dict'])
            optimizers[0].load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    cudnn.benchmark = True

    get_train_loader = get_pytorch_train_loader
    get_val_loader = get_pytorch_val_loader

    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size,
                                                      custom_oi_path=args.oi_list, oi_thresh=args.oi_thresh, workers=args.workers,
                                                      input_size=args.input_size)
    val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)

    if args.evaluate:
        from collections import OrderedDict
        if os.path.isfile(args.weight):
            print("=> loading pretrained weight '{}'".format(args.weight))
            source_state = torch.load(args.weight)
            if 'state_dict' in source_state:
                source_state = source_state['state_dict']
            if 'model' in source_state:
                source_state = source_state['model']
            target_state = OrderedDict()
            for k, v in source_state.items():
                #if k.startswith('module.attacker.model.'):
                #    k = k[len('module.attacker.model.'):]
                #else:
                #    continue
                if k[:7] != 'module.':
                    k = 'module.' + k
                target_state[k] = v
            models[0].load_state_dict(target_state, strict=True)
        else:
            print("=> no weight found at '{}'".format(args.weight))

        validate(val_loader, val_loader_len, models[0], criterion, adv_eps=args.adv_eps,
                 euclidean_adv=args.euclidean)
        return

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        print(f'\nEpoch: [{epoch + 1} | {args.epochs}]')

        # train for one epoch
        train_losses, train_accs = train(train_loader, train_loader_len, models, criterion, optimizers, epoch)

        # evaluate on validation set
        val_loss, prec1, prec5, adv_prec1, adv_prec5 = validate(val_loader, val_loader_len,
                                                                models[0], criterion, adv_eps=args.adv_eps,
                                                                euclidean_adv=args.euclidean)

        lr = optimizers[0].param_groups[0]['lr']

        # append logger file
        logger.append([lr, train_losses[0], val_loss, train_accs[0], prec1])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_losses[0], 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_accs[0], 'validation accuracy': prec1}, epoch + 1)

        if len(train_losses) > 1:
            writer.add_scalars('loss_aux', {'train loss': train_losses[1], 'validation loss': val_loss}, epoch + 1)
            writer.add_scalars('accuracy_aux', {'train accuracy': train_accs[1], 'validation accuracy': prec1}, epoch + 1)
        print(f'Val results: {prec1:.2f} ; {prec5:.2f} ; {adv_prec1:.2f} ; {adv_prec5:.2f}')

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': models[0].state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizers[0].state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        if (epoch + 1) % 10 == 0 and len(models) > 1:
            val_loss, prec1, prec5, adv_prec1, adv_prec5 = validate(val_loader, val_loader_len,
                                                                    models[1], criterion, adv_eps=args.adv_eps,
                                                                    euclidean_adv=args.euclidean)
            print(f'Additional model val results: {prec1:.2f} ; {prec5:.2f} ; {adv_prec1:.2f} ; {adv_prec5:.2f}')

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()

    print('Best accuracy:')
    print(best_prec1)

def kl_div(input, target):
    return nn.functional.kl_div(nn.functional.log_softmax(input, dim=1),
                                nn.functional.softmax(target.detach(), dim=1), reduction='batchmean')
    #t_sm = nn.functional.softmax(target.detach(), dim=1)
    #i_sm = nn.functional.softmax(input, dim=1)
    #return torch.sum(t_sm*torch.log(t_sm / (i_sm + eps) + eps), dim=1).mean()

def train(train_loader, train_loader_len, models, criterion, optimizers, epoch):
    bar = Bar('Processing', max=train_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_losses = [AverageMeter() for _ in models]
    top1_all = [AverageMeter() for _ in models]
    top5_all = [AverageMeter() for _ in models]

    # switch to train mode
    for model in models:
        model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        for optimizer in optimizers:
            adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        outputs = []
        for model in models:
            if args.adv_eps > 0.:
                model.eval()
                adv_data = pgd_attack(model, input, target, epsilon=args.adv_eps, euclidean=args.euclidean,
                                    step_size=2./3.*args.adv_eps, criterion=criterion)
                model.train()
                outputs.append(model(adv_data))
            else:
                outputs.append(model(input))

        losses = [criterion(output, target) for output in outputs]

        if len(models) > 1:
            losses[0] += kl_div(outputs[0], outputs[1])
            losses[1] += kl_div(outputs[1], outputs[0])

        # measure accuracy and record loss
        for i, output in enumerate(outputs):
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            avg_losses[i].update(losses[i].item(), input.size(0))
            top1_all[i].update(prec1.item(), input.size(0))
            top5_all[i].update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        for optimizer in optimizers:
            optimizer.zero_grad()
        for loss in losses:
            loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=i + 1,
                    size=train_loader_len,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=avg_losses[0].avg,
                    top1=top1_all[0].avg,
                    top5=top5_all[0].avg,
                    )
        bar.next()
    bar.finish()
    return ([loss.avg for loss in avg_losses], [top1.avg for top1 in top1_all])


def validate(val_loader, val_loader_len, model, criterion, adv_eps=0.0, euclidean_adv=False):
    bar = Bar('Processing', max=val_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        if adv_eps > 0.:
            adv_data = pgd_attack(model, input, target, epsilon=adv_eps, euclidean=euclidean_adv,
                                  step_size=2./3.*adv_eps, criterion=criterion)
            with torch.no_grad():
                output = model(adv_data)
                loss = criterion(output, target)
            adv_prec1, adv_prec5 = accuracy(output, target, topk=(1, 5))
            adv_top1.update(adv_prec1.item(), input.size(0))
            adv_top5.update(adv_prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=i + 1,
                    size=val_loader_len,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        if adv_eps > 0.:
            bar.suffix += f'| adv_top1 {adv_top1.avg:.4f} | adv_top5 {adv_top5.avg:.4f}'
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg, adv_top1.avg, adv_top5.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
