import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import sys
import torch.distributed as dist
import timm
import math
import time
import argparse
import json
from .cse_evaluater import test
import utils.lr_sched as lr_sched
import timm.optim.optim_factory as optim_factory
import ctypes
import model.cse_model
from torch.utils.data import DataLoader
import datetime
from progress.bar import Bar

torch.manual_seed(1)
torch.cuda.manual_seed(1)
from utils import misc
# import utils.misc as misc
from reid_cse.cse_joint_db import CSEDataset
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import FP32ScalerWithGradNormCount as FP32Scaler


def get_args_parser():
    parser = argparse.ArgumentParser('Continual Surface Embedding')
    # reid_cse
    parser.add_argument('--repeat', default=1, type=int)
    # others
    parser.add_argument('--ckpt_dir', default='./checkpoint', type=str)
    parser.add_argument('--finetune', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--eval', default=None, type=str)
    # train
    parser.add_argument('--weight_decay', default=5e-2, type=float)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--blr', default=1e-3, type=int)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--max_epochs', default=70, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--layer_decay', default=1.0, type=int)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--fp32', action='store_true', default=True)
    parser.add_argument('--max_norm', default=None, type=float)
    parser.add_argument('--save_checkpoint_each_epoch', default=5, type=int)
    # distributed training parameters
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--world_size', type=int, default=1)
    # parser.add_argument('--init_method', type=str, default='env://')
    return parser


def configure_file_list(args):
    train_json_list = [
        './data/reid_cse/DP3D_train.json',
        './data/reid_cse/LTCC_train.json',
        './data/reid_cse/Market1501_train.json',
        './data/reid_cse/PRCC_train.json',
        './data/reid_cse/VC_Clothes_train.json',
        './data/reid_cse/densepose_coco_2014_train.json'
    ]
    test_json_list = [
        './data/reid_cse/DP3D_test.json',
        './data/reid_cse/LTCC_test.json',
        './data/reid_cse/Market1501_test.json',
        './data/reid_cse/PRCC_test.json',
        './data/reid_cse/VC_Clothes_test.json',
        './data/reid_cse/densepose_coco_2014_test.json',
    ]
    args.dataset_weights = [5]
    args.dataset_weights = [x / sum(args.dataset_weights) for x in args.dataset_weights]
    return train_json_list, test_json_list


def train_epoch(model, optimizer, train_loader, loss_scaler, epoch, args):
    bar = Bar(f'Epoch {epoch + 1}/{args.max_epochs}', fill='#', max=len(train_loader))

    model.train(True)
    optimizer.zero_grad()
    total_cse_loss = 0.0
    total_dp_masks_loss = 0.0
    total_loss = 0.0
    total_acc = 0.0
    for data_iter_step, samples in enumerate(train_loader):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)
        # print(samples.keys())
        # print(samples['img'].shape)
        for k, v in samples.items():
            if isinstance(v, torch.Tensor):
                samples[k] = samples[k].to('cuda')
        losses, acc = model(**samples, mode='loss')
        loss_cse = losses['loss_cse'].item()
        loss_dp_masks = losses['loss_dp_masks'].item()
        loss = sum(losses.values())
        loss_value = loss.item()
        total_loss += loss_value
        total_cse_loss += loss_cse
        total_dp_masks_loss += loss_dp_masks
        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)
        grad_norm = loss_scaler(loss, optimizer, clip_grad=args.max_norm, parameters=model.parameters(),
                                update_grad=True)
        # print(grad_norm)
        optimizer.zero_grad()
        lr = optimizer.param_groups[-1]['lr']
        summary_string = f'({data_iter_step + 1}/{len(train_loader)}) | Total: {bar.elapsed_td} | ' \
                         f'ETA: {bar.eta_td:} | loss: {total_loss / (data_iter_step + 1):.4f} | ' \
                         f'loss_cse: {total_cse_loss / (data_iter_step + 1):.4f} | ' \
                         f'loss_dp_masks: {total_dp_masks_loss / (data_iter_step + 1):.4f} ' \
                         f'|acc: {acc:.4f} | lr: {lr}'
        bar.suffix = summary_string
        bar.next()


def main(args):
    cudnn.benchmark = False
    args.lr = args.blr * args.batch_size / 256.0
    print(args.lr)
    model = timm.create_model('cse_darknet19_binary2')
    model = model.cuda()
    if args.finetune is not None:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print('Load pre-trained checkpoint from: %s' % args.finetune)
        for k in ['reid_cse', 'module', 'encoder']:
            if k in checkpoint:
                checkpoint = checkpoint[k]
                break
        new_checkpoint = dict()
        for k, v in checkpoint.item():
            if 'module.' in k:
                k = k.replace('module.', '')
            new_checkpoint[k] = v
        # load pretrained reid_cse
        msg = model.load_state_dict(new_checkpoint, strict=True)
        print(msg)
    train_json_list, test_json_list = configure_file_list(args)
    train_set = CSEDataset(train_json_list, train=True, repeat=args.repeat)
    test_sets = [CSEDataset([x]) for x in test_json_list]

    def collate_fn(batch):
        batch_dict = {}
        list_names = ['dp_x', 'dp_y', 'dp_I', 'dp_U', 'dp_V']
        for k in batch[0].keys():
            if k in list_names:
                batch_dict[k] = [x[k] for x in batch]

            else:
                batch_dict[k] = torch.stack([x[k] for x in batch], 0)
        return batch_dict

    # initialize train loader
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        shuffle=True,
    )
    test_loaders = [DataLoader(
        test_set, batch_size=1, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, shuffle=False,
    ) for test_set in test_sets]

    model_ema = None
    print(args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)
    loss_scaler = NativeScaler() if not args.fp32 else FP32Scaler()
    start_epoch = 0
    if args.resume is not None:
        start_epoch = misc.load_model(resume=args.resume, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
    else:
        start_epoch = misc.auto_load_model(model=model, optimizer=optimizer, ckpt_path=args.ckpt_dir,
                                           loss_scaler=loss_scaler)
    if args.eval:
        test(model, test_loaders, test_json_list, args.ckpt_dir, args, epoch=0)
    print(f'Start training for {start_epoch} epochs')
    start_time = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        train_stats = train_epoch(
            model, optimizer, train_loader, loss_scaler, epoch, args
        )
        misc.save_model(
            args.ckpt_dir, model=model, model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
            name='latest'
        )
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        # f = open(osp.join(args.ckpt_dir, "log.txt"), mode='a', encoding='utf-8')
        # f.write(json.dumps(log_stats) + '\n')
        if ((epoch + 1) % args.save_checkpoint_each_epoch == 0 or epoch + 1 == args.max_epochs):
            misc.save_model(
                args.ckpt_dir, model=model, model_ema=model_ema, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )
            test(model, test_loaders, test_json_list, args.ckpt_dir, args, epoch=epoch)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
