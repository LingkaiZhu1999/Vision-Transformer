# https://github.com/pytorch/examples/blob/main/imagenet/main.py
# adapted for vision transformer and webdataset
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import default_collate
from torch.utils.data import Subset
from torch.distributed.optim import ZeroRedundancyOptimizer

import webdataset as wds
import wandb

from transformer import Transformer_VM
from utils import learning_rate_schedule

torch.set_float32_matmul_precision('high')

model_names = sorted(name for name in models.__dict__
 
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='ViT')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--d_model", type=int, default=768)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--d_ff", type=int, default=3072)
parser.add_argument("--num_classes", type=int, default=1000)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument("--early_stopping_patience", type=int, default=15)
parser.add_argument("--min_delta", type=float, default=5e-4)
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
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--t_warm_up", type=int, default=10)
parser.add_argument("--t_cos_anneal", type=int, default=120)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--dropout', default=0.0, type=float, metavar='D',
                    help='dropout rate (default: 0.0)')
parser.add_argument('--randaugment-num-ops', default=2, type=int,
                    help='number of RandAugment ops to apply per image')
parser.add_argument('--randaugment-magnitude', default=9, type=int,
                    help='RandAugment magnitude')
parser.add_argument('--color-jitter', default=0.4, type=float,
                    help='strength for ColorJitter augmentation')
parser.add_argument('--random-erase-prob', default=0.25, type=float,
                    help='probability for RandomErasing')
parser.add_argument('--mixup-alpha', default=0.8, type=float,
                    help='mixup beta distribution alpha; set 0 to disable')
parser.add_argument('--cutmix-alpha', default=1.0, type=float,
                    help='cutmix beta distribution alpha; set 0 to disable')
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--no-accel', action='store_true',
                    help='disables accelerator')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--compile', action='store_true', help="use torch.compile to compile the model")
parser.add_argument('--bf16', action='store_true', help="use bfloat16 precision for training")
parser.add_argument('--use_zero', action='store_true', help="use zero optimizer from deepspeed")

best_acc1 = 0
IMAGENET_TRAIN_SAMPLES = 1281167
IMAGENET_VAL_SAMPLES = 50000
NUM_CLASSES = 1000


class MixAugmentCollate:
    """Pickle-safe collate wrapper for mixup/cutmix augmentation."""

    def __init__(self, alpha_cutmix, alpha_mixup, num_classes):
        cutmix = v2.CutMix(alpha=alpha_cutmix, num_classes=num_classes)
        mixup = v2.MixUp(alpha=alpha_mixup, num_classes=num_classes)
        self.transform = v2.RandomChoice([cutmix, mixup])

    def __call__(self, batch):
        return self.transform(*default_collate(batch))


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        effective_world_size = torch.accelerator.device_count() * args.world_size
    elif args.distributed and args.world_size > 0:
        effective_world_size = args.world_size
    else:
        effective_world_size = 1

    args.train_batches = IMAGENET_TRAIN_SAMPLES // (args.batch_size * effective_world_size)
    args.val_batches = IMAGENET_VAL_SAMPLES // (args.batch_size * effective_world_size)
    args.total_steps = args.epochs * args.train_batches

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type =='cuda':
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def label_to_index(label):
    return int(label) 

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        if args.gpu is not None:
            torch.accelerator.set_device_index(args.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        is_main_process = not args.distributed or (args.distributed and args.rank == 0)
    
        if is_main_process:
            run = wandb.init(project="ViT", config=args)
        else:
            run = None
    # create model
    if args.pretrained:
        import transformers
        print("=> using pre-trained model 'google/vit-base-patch16-224-in21k'")
        model = transformers.ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    else:
        print("=> creating Vision Transformer model")
        model = Transformer_VM(
            image_size=args.image_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_classes=args.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    print_peak_memory("Max memory allocated after creating local model", device)

    if args.compile:
        print("compiling the model with torch.compile...")
        model = torch.compile(model)

    if not use_accel:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if device.type == 'cuda':
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(device)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)

    print_peak_memory("Max memory allocated after creating DDP", device)


    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    if args.use_zero:
        optimizer = ZeroRedundancyOptimizer(model.parameters(), 
                    optimizer_class=torch.optim.AdamW, 
                    lr=args.lr, 
                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'{device.type}:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(IMAGENET_TRAIN_SAMPLES, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(IMAGENET_VAL_SAMPLES, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        # traindir = os.path.join(args.data, 'train')
        # valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        world_size = args.world_size if args.distributed else 1
        num_workers = max(1, args.workers) 
        train_samples_per_worker = IMAGENET_TRAIN_SAMPLES // (world_size * num_workers)
        val_samples_per_worker = IMAGENET_VAL_SAMPLES // (world_size * num_workers)

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                args.image_size,
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(
                num_ops=args.randaugment_num_ops,
                magnitude=args.randaugment_magnitude,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ColorJitter(
                brightness=args.color_jitter,
                contrast=args.color_jitter,
                saturation=args.color_jitter,
                hue=min(0.1, args.color_jitter / 4),
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=args.random_erase_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = wds.WebDataset(args.data + "imagenet1k-train-{0000..1023}.tar",
                                       nodesplitter=wds.split_by_node,
                                       workersplitter=wds.split_by_worker,
                                       shardshuffle=1024).\
            shuffle(1000).decode("pil").to_tuple("jpg", "cls").map_tuple(
             train_transforms, label_to_index
        ).with_epoch(train_samples_per_worker) 

        # train_dataset = datasets.ImageFolder(
        #     traindir,
        #     transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
        val_dataset = wds.WebDataset(args.data + "imagenet1k-validation-{00..63}.tar",
                                    nodesplitter=wds.split_by_node,
                                    workersplitter=wds.split_by_worker,
                                    shardshuffle=False).\
            decode("pil").to_tuple("jpg", "cls").map_tuple(
                val_transforms, label_to_index
            ).with_epoch(val_samples_per_worker) 
        # val_dataset = datasets.ImageFolder(
        #     valdir,
        #     transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    # else:
    train_sampler = None
    val_sampler = None

    use_mix_augment = args.mixup_alpha > 0 or args.cutmix_alpha > 0
    if use_mix_augment:
        train_collate_fn = MixAugmentCollate(
            alpha_cutmix=args.cutmix_alpha,
            alpha_mixup=args.mixup_alpha,
            num_classes=args.num_classes,
        )
    else:
        train_collate_fn = default_collate

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True, persistent_workers=True, collate_fn=train_collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        persistent_workers=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)

        # train for one epoch
        loss, train_acc1, train_acc5, current_lr = train(
            train_loader, model, criterion, optimizer, epoch, device, args
        )
        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
        if run is not None:
            log_data = {
                "train/loss": float(loss), 
                "train/lr": float(current_lr),
                "val/acc1": float(acc1), 
                "val/acc5": float(acc5),
                "epoch": epoch,
            }
            if train_acc1 is not None and train_acc5 is not None:
                log_data["train/acc1"] = float(train_acc1)
                log_data["train/acc5"] = float(train_acc5)
            run.log(log_data)

        # scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # Fix for ZeroRedundancyOptimizer: consolidate state on all ranks, save only on rank 0
        if args.use_zero and args.distributed:
            optimizer.consolidate_state_dict(to=0)
            if args.rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict()
                }, is_best)
        else:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict()
                }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    
    use_accel = not args.no_accel and torch.accelerator.is_available()
    compute_train_accuracy = args.mixup_alpha <= 0 and args.cutmix_alpha <= 0

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    if compute_train_accuracy:
        top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.NONE)
        top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.NONE)
        progress_meters = [batch_time, data_time, losses, top1, top5]
    else:
        progress_meters = [batch_time, data_time, losses]
        top1 = None
        top5 = None

    progress = ProgressMeter(
        args.train_batches,
        progress_meters,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):

        current_lr = learning_rate_schedule(
            t=epoch * args.train_batches + i,
            lr_max=args.lr,
            lr_min=args.min_lr,
            t_warm_up=args.t_warm_up,
            total_steps=args.total_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images, target = batch

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if args.bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        losses.update(loss.item(), images.size(0))
        if compute_train_accuracy:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print_peak_memory("Max memory allocated before optimizer step()", device)
        optimizer.step()
        print_peak_memory("Max memory allocated after optimizer step()", device)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    if compute_train_accuracy:
        return losses.avg, top1.avg, top5.avg, current_lr
    return losses.avg, None, None, current_lr

def validate(val_loader, model, criterion, args):

    use_accel = not args.no_accel and torch.accelerator.is_available()

    def run_validate(loader, base_progress=0):

        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if use_accel:
                    if args.gpu is not None and device.type=='cuda':
                        torch.accelerator.set_device_index(args.gpu)
                        images = images.cuda(args.gpu, non_blocking=True)
                        target = target.cuda(args.gpu, non_blocking=True)
                    else:
                        images = images.to(device)
                        target = target.to(device)

                # compute output
                if args.bf16:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        output = model(images)
                        loss = criterion(output, target)
                else:
                    output = model(images)
                    loss = criterion(output, target)
                # output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.AVERAGE)
    
    world_size = args.world_size if args.distributed else 1
    progress = ProgressMeter(
        args.val_batches,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    # if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
    #     aux_val_dataset = Subset(val_loader.dataset,
    #                              range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
    #     aux_val_loader = torch.utils.data.DataLoader(
    #         aux_val_dataset, batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.workers, pin_memory=True)
    #     run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):    
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
