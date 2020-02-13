## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
from argparser import argparser
import os
import sys
import copy
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
import numpy as np
from datasets import loaders
from model_defs import Flatten, model_mlp_any, model_cnn_1layer, model_cnn_2layer, model_cnn_4layer, model_cnn_3layer
from bound_layers import BoundSequential, BoundLinear, BoundConv2d, ParallelBound, ParallelBoundPool
from attacks.patch_attacker import PatchAttacker
from attacks.pgd_attacker import PGDAttacker
#from attacks.debug import PGDAttacker
import torch.optim as optim
# from gpu_profile import gpu_profile
import time
from datetime import datetime
import torch.nn as nn
from config import load_config, get_path, config_modelloader, config_dataloader, update_dict
import torch
from PIL import Image
from matplotlib import pyplot as plt
from unet import ResNetUNet
from itertools import chain
from tqdm import tqdm
import pdb
# sys.settrace(gpu_profile)

torch.backends.cudnn.benchmark=True

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

class Logger(object):
    def __init__(self, log_file = None):
        self.log_file = log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file = self.log_file)
            self.log_file.flush()


def Train(model, model_id, t, loader, start_eps, end_eps, max_eps, norm, logger, verbose, train, opt, method, adv_net=None, unetopt=None, **kwargs):
    # if train=True, use training mode
    # if train=False, use test mode, no back prop
    num_class = 10
    losses = AverageMeter()
    unetlosses = AverageMeter()
    unetloss = None
    errors = AverageMeter()
    adv_errors = AverageMeter()
    robust_errors = AverageMeter()
    regular_ce_losses = AverageMeter()
    adv_ce_losses = AverageMeter()
    robust_ce_losses = AverageMeter()
    batch_time = AverageMeter()
    # initial 
    kappa = 1
    factor = 1
    if train:
        model.train()
        if adv_net is not None:
            adv_net.train()
    else:
        model.eval()
        if adv_net is not None:
            adv_net.eval()
    # pregenerate the array for specifications, will be used for scatter
    if method == "robust":
        sa = np.zeros((num_class, num_class - 1), dtype = np.int32)
        for i in range(sa.shape[0]):
            for j in range(sa.shape[1]):
                if j < i:
                    sa[i][j] = j
                else:
                    sa[i][j] = j + 1
        sa = torch.LongTensor(sa)
    elif method == "adv":
        if kwargs["attack_type"] == "patch-random":
            attacker = PatchAttacker(model, loader.mean, loader.std, kwargs)
        elif kwargs["attack_type"] == "patch-strong":
            attacker = PatchAttacker(model, loader.mean, loader.std, kwargs)
        elif kwargs["attack_type"] == "PGD":
            attacker = PGDAttacker(model, loader.mean, loader.std, kwargs)
    total = len(loader.dataset)
    batch_size = loader.batch_size
    if train:
        batch_eps = np.linspace(start_eps, end_eps, total// (batch_size*args.grad_acc_steps) + 1)
        batch_eps = batch_eps.repeat(args.grad_acc_steps)
    else:
        batch_eps = np.linspace(start_eps, end_eps, total // (batch_size) + 1)

    if end_eps < 1e-6:
        logger.log('eps {} close to 0, using natural training'.format(end_eps))
        method = "natural"

    if train:
        iterator = enumerate(loader)
    else:
        iterator = tqdm(enumerate(loader))
    if train:
        opt.zero_grad()
        if unetopt is not None:
            unetopt.zero_grad()
    for i, (data, labels) in iterator:
        if "sample_limit" in kwargs and i*loader.batch_size > kwargs["sample_limit"]:
            break
        start = time.time()
        eps = batch_eps[i]

        if method == "robust":
            # generate specifications
            c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
            # remove specifications to self
            I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
            c = (c[I].view(data.size(0),num_class-1,num_class))
            # scatter matrix to avoid computing margin to self
            sa_labels = sa[labels]
            # storing computed lower bounds after scatter
            lb_s = torch.zeros(data.size(0), num_class)

            #calculating upper and lower bound of the input
            if len(loader.std) == 1:
                std = torch.tensor([loader.std], dtype=torch.float)[:, None, None]
                mean = torch.tensor([loader.mean], dtype=torch.float)[:, None, None]
            elif len(loader.std) == 3:
                std = torch.tensor(loader.std, dtype=torch.float)[None, :, None, None]
                mean = torch.tensor(loader.mean, dtype=torch.float)[None, :, None, None]
            if kwargs["bound_type"] == "sparse-interval":
                data_ub = data
                data_lb = data
                eps = (eps / std).max()
            else:
                data_ub = (data + eps/std)
                data_lb = (data - eps/std)
                ub = ((1 - mean) / std)
                lb = (-mean / std)
                data_ub = torch.min(data_ub, ub)
                data_lb = torch.max(data_lb, lb)

            if list(model.parameters())[0].is_cuda:
                data_ub = data_ub.cuda()
                data_lb = data_lb.cuda()
                c = c.cuda()
                sa_labels = sa_labels.cuda()
                lb_s = lb_s.cuda()

        if list(model.parameters())[0].is_cuda:
            data = data.cuda()
            labels = labels.cuda()
        # the regular cross entropy
        if torch.cuda.device_count()>1:
            output = nn.DataParallel(model)(data)
        else:
            output = model(data)

        regular_ce = CrossEntropyLoss()(output, labels)
        regular_ce_losses.update(regular_ce.cpu().detach().numpy(), data.size(0))
        errors.update(torch.sum(torch.argmax(output, dim=1)!=labels).cpu().detach().numpy()/data.size(0), data.size(0))

        # the adversarial cross entropy
        if method == "adv":
            if kwargs["attack_type"]=="PGD":
                data_adv = attacker.perturb(data, labels, norm)
            elif kwargs["attack_type"]=="patch-random":
                data_adv = attacker.perturb(data, labels, norm, random_count=kwargs["random_mask_count"])
            else:
                raise RuntimeError("Unknown attack_type " + kwargs["bound_type"])
            output_adv = model(data_adv)
            adv_ce = CrossEntropyLoss()(output_adv, labels)
            adv_ce_losses.update(adv_ce.cpu().detach().numpy(), data.size(0))
            adv_errors.update(
                torch.sum(torch.argmax(output_adv, dim=1) != labels).cpu().detach().numpy() / data.size(0),
                data.size(0))


        if verbose or method == "robust":
            if kwargs["bound_type"] == "interval":
                ub, lb = model.interval_range(x_U=data_ub, x_L=data_lb, eps=eps, C=c)
            elif kwargs["bound_type"] == "sparse-interval":
                ub, lb = model.interval_range(x_U=data_ub, x_L=data_lb, eps=eps, C=c, k=kwargs["k"], Sparse=True)
            elif kwargs["bound_type"] == "patch-interval":
                if kwargs["attack_type"] == "patch-all" or kwargs["attack_type"] == "patch-all-pool":
                    if kwargs["attack_type"] == "patch-all":
                        width = data.shape[2] - kwargs["patch_w"] + 1
                        length = data.shape[3] - kwargs["patch_l"] + 1
                        pos_patch_count = width * length
                        final_bound_count = pos_patch_count
                    elif kwargs["attack_type"] == "patch-all-pool":
                        width = data.shape[2] - kwargs["patch_w"] + 1
                        length = data.shape[3] - kwargs["patch_l"] + 1
                        pos_patch_count = width * length
                        final_width = width
                        final_length = length
                        for neighbor in kwargs["neighbor"]:
                            final_width = ((final_width - 1) // neighbor + 1)
                            final_length = ((final_length - 1) // neighbor + 1)
                        final_bound_count = final_width * final_length

                    patch_idx = torch.arange(pos_patch_count, dtype=torch.long)[None, :]
                    if kwargs["attack_type"] == "patch-all" or kwargs["attack_type"] == "patch-all-pool":
                        x_cord = torch.zeros((1, pos_patch_count), dtype=torch.long)
                        y_cord = torch.zeros((1, pos_patch_count), dtype=torch.long)
                        idx = 0
                        for w in range(width):
                            for l in range(length):
                                x_cord[0, idx] = w
                                y_cord[0, idx] = l
                                idx = idx + 1

                    # expand the list to include coordinates from the complete patch
                    patch_idx = [patch_idx.flatten()]
                    x_cord = [x_cord.flatten()]
                    y_cord = [y_cord.flatten()]
                    for w in range(kwargs["patch_w"]):
                        for l in range(kwargs["patch_l"]):
                            patch_idx.append(patch_idx[0])
                            x_cord.append(x_cord[0] + w)
                            y_cord.append(y_cord[0] + l)

                    patch_idx = torch.cat(patch_idx, dim=0)
                    x_cord = torch.cat(x_cord, dim=0)
                    y_cord = torch.cat(y_cord, dim=0)

                    # create masks for each data point
                    mask = torch.zeros([1, pos_patch_count, data.shape[2], data.shape[3]],
                                       dtype=torch.uint8)
                    mask[:, patch_idx, x_cord, y_cord] = 1
                    mask = mask[:, :, None, :, :]
                    mask = mask.cuda()
                    data_ub = torch.where(mask, data_ub[:, None, :, :, :], data[:, None, :, :, :])
                    data_lb = torch.where(mask, data_lb[:, None, :, :, :], data[:, None, :, :, :])

                    # data_ub size (#data*#possible patches, #channels, width, length)
                    data_ub = data_ub.view(-1, *data_ub.shape[2:])
                    data_lb = data_lb.view(-1, *data_lb.shape[2:])

                    c = c.repeat_interleave(final_bound_count, dim=0)

                elif kwargs["attack_type"] == "patch-random" or kwargs["attack_type"] == "patch-nn":
                    # First calculate the number of considered patches
                    if kwargs["attack_type"] == "patch-random":
                        pos_patch_count = kwargs["patch_count"]
                        final_bound_count = pos_patch_count
                        c = c.repeat_interleave(pos_patch_count, dim=0)
                    elif kwargs["attack_type"] == "patch-nn":
                        class_count = 10
                        pos_patch_count = kwargs["patch_count"] * class_count
                        final_bound_count = pos_patch_count
                        c = c.repeat_interleave(pos_patch_count, dim=0)


                    # Create four lists that enumerate the coordinate of the top left corner of the patch
                    # patch_idx, data_idx, x_cord, y_cord shpe = (# of datapoints, # of possible patches)
                    patch_idx = torch.arange(pos_patch_count, dtype=torch.long)[None, :].repeat(data.shape[0], 1)
                    data_idx = torch.arange(data.shape[0], dtype=torch.long)[:, None].repeat(1, pos_patch_count)
                    if kwargs["attack_type"] == "patch-random":
                        x_cord = torch.randint(0, data.shape[2] - kwargs["patch_w"]+1, (data.shape[0], pos_patch_count))
                        y_cord = torch.randint(0, data.shape[3] - kwargs["patch_l"]+1, (data.shape[0], pos_patch_count))
                    elif kwargs["attack_type"] == "patch-nn":
                        lbs_pred = adv_net(data)
                        # Take only the feasible location
                        lbs_pred = lbs_pred[:, :,
                                   0:lbs_pred.size(2) - kwargs["patch_l"] + 1,
                                   0:lbs_pred.size(3) - kwargs["patch_w"] + 1]

                        lbs_pred = lbs_pred.reshape(lbs_pred.size(0) * lbs_pred.size(1), -1)
                        # lbs_pred (# datapoints*# of classes, #flattened image dim)
                        select_prob = nn.Softmax(1)(-lbs_pred * kwargs["T"])
                        # select_prob (# datapoints*# of classes, #flattened image dim)
                        random_loc = torch.multinomial(select_prob, kwargs["patch_count"], replacement=False)
                        # random_loc (# datapoints*# of classes, patch_count)
                        random_loc = random_loc.view(data.size(0), -1)
                        # random_loc (# datapoints, # of classes*patch_count)

                        x_cord = random_loc % (data.size(3) - kwargs["patch_w"] + 1)
                        y_cord = random_loc // (data.size(2) - kwargs["patch_l"] + 1)

                    # expand the list to include coordinates from the complete patch
                    patch_idx = [patch_idx.flatten()]
                    data_idx = [data_idx.flatten()]
                    x_cord = [x_cord.flatten()]
                    y_cord = [y_cord.flatten()]
                    for w in range(kwargs["patch_w"]):
                        for l in range(kwargs["patch_l"]):
                            patch_idx.append(patch_idx[0])
                            data_idx.append(data_idx[0])
                            x_cord.append(x_cord[0]+w)
                            y_cord.append(y_cord[0]+l)

                    patch_idx = torch.cat(patch_idx, dim=0)
                    data_idx = torch.cat(data_idx, dim=0)
                    x_cord = torch.cat(x_cord, dim=0)
                    y_cord = torch.cat(y_cord, dim=0)

                    #create masks for each data point
                    mask = torch.zeros([data.shape[0], pos_patch_count, data.shape[2], data.shape[3]],
                                       dtype=torch.uint8)
                    mask[data_idx, patch_idx, x_cord, y_cord] = 1
                    mask = mask[:, :, None, :, :]
                    mask = mask.cuda()
                    data_ub = torch.where(mask, data_ub[:, None, :, :, :], data[:, None, :, :, :])
                    data_lb = torch.where(mask, data_lb[:, None, :, :, :], data[:, None, :, :, :])

                    # data_ub size (#data*#possible patches, #channels, width, length)
                    data_ub = data_ub.view(-1, *data_ub.shape[2:])
                    data_lb = data_lb.view(-1, *data_lb.shape[2:])

                # forward pass all bounds
                if torch.cuda.device_count() > 1:
                    if kwargs["attack_type"] == "patch-all-pool":
                        ub, lb = nn.DataParallel(ParallelBoundPool(model))(x_U=data_ub, x_L=data_lb, eps=eps, C=c,
                                                                          neighbor=kwargs["neighbor"],
                                                                          pos_patch_width=width, pos_patch_length=length)
                    else:
                        ub, lb = nn.DataParallel(ParallelBound(model))(x_U=data_ub, x_L=data_lb,
                                                                       eps=eps, C=c)
                else:
                    if kwargs["attack_type"] == "patch-all-pool":
                        ub, lb = model.interval_range_pool(x_U=data_ub, x_L=data_lb, eps=eps, C=c,
                                                      neighbor=kwargs["neighbor"],
                                                      pos_patch_width=width, pos_patch_length=length)
                    else:
                        ub, lb = model.interval_range(x_U=data_ub, x_L=data_lb, eps=eps, C=c)

                # calculate unet loss
                if kwargs["attack_type"] == "patch-nn":
                    labels_mod = labels.repeat_interleave(pos_patch_count, dim=0)
                    sa_labels_mod = sa[labels_mod]
                    sa_labels_mod = sa_labels_mod.cuda()
                    # storing computed lower bounds after scatter
                    lb_s_mod = torch.zeros(data.size(0) * pos_patch_count, num_class).cuda()
                    lbs_actual = lb_s_mod.scatter(1, sa_labels_mod, lb)
                    # lbs_actual (# data * # of logits * # of classes, # of classes)

                    # lbs_pred (# datapoints*# of logits, #flattened image dim)
                    lbs_pred = lbs_pred.view(data.shape[0], num_class, -1)
                    # lbs_pred (# datapoints, # of logits, #flattened image dim)
                    lbs_pred = lbs_pred.permute(0, 2, 1)
                    # lbs_pred (# datapoints, #flattened image dim, # of logits)

                    # random_loc (# datapoints, # of logits*patch_count)
                    random_loc = random_loc.unsqueeze(2)
                    random_loc = random_loc.repeat_interleave(10, dim=2)
                    lbs_pred = lbs_pred.gather(1, random_loc)
                    # lbs_pred (# datapoints, # of logits*patch_count, # of logits)
                    lbs_pred = lbs_pred.view(-1, num_class)
                    # lbs_pred (# datapoints*# of logits*patch_count, # of logits)
                    unetloss = nn.MSELoss()(lbs_actual.detach(), lbs_pred)

                lb = lb.reshape(-1, final_bound_count, lb.shape[1])
                lb = torch.min(lb, dim=1)[0]
            else:
                raise RuntimeError("Unknown bound_type " + kwargs["bound_type"])
            # pdb.set_trace()
            lb = lb_s.scatter(1, sa_labels, lb)
            robust_ce = CrossEntropyLoss()(-lb, labels)

        if method == "robust":
            loss = robust_ce
        elif method == "natural":
            loss = regular_ce
        elif method == "adv":
            loss = adv_ce
        elif method == "robust_natural":
            natural_final_factor = kwargs["final-kappa"]
            kappa = (max_eps - eps * (1.0 - natural_final_factor)) / max_eps
            loss = (1-kappa) * robust_ce + kappa * regular_ce
        else:
            raise ValueError("Unknown method " + method)

        if train:
            if unetloss is not None:
                unetloss.backward()
                unetlosses.update(unetloss.cpu().detach().numpy(), data.size(0))
            loss = loss
            loss.backward()
            if (i + 1) % args.grad_acc_steps == 0 or i == len(loader) - 1:
                if unetloss is not None:
                    for p in adv_net.parameters():
                        p.grad /= args.grad_acc_steps
                    unetopt.step()
                for p in model.parameters():
                    p.grad /= args.grad_acc_steps
                opt.step()
                opt.zero_grad()

        batch_time.update(time.time() - start)


        losses.update(loss.cpu().detach().numpy(), data.size(0))

        if verbose or method == "robust":
            robust_ce_losses.update(robust_ce.cpu().detach().numpy(), data.size(0))
            robust_errors.update(torch.sum((lb<0).any(dim=1)).cpu().detach().numpy() / data.size(0), data.size(0))
        if i % 50 == 0 and train:
            logger.log(  '[{:2d}:{:4d}]: eps {:4f}  '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
                    'Unet Loss {unetloss.val:.4f} ({unetloss.avg:.4f})  '
                    'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
                    'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
                    'ACE {adv_ce_loss.val:.4f} ({adv_ce_loss.avg:.4f})  '
                    'Err {errors.val:.4f} ({errors.avg:.4f})  '
                    'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
                    'Adv Err {adv_errors.val:.4f} ({adv_errors.avg:.4f})  '
                    'beta {factor:.3f} ({factor:.3f})  '
                    'kappa {kappa:.3f} ({kappa:.3f})  '.format(
                    t, i, eps, batch_time=batch_time,
                    loss=losses, unetloss=unetlosses, errors=errors, robust_errors = robust_errors, adv_errors = adv_errors,
                    regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses,
                    adv_ce_loss = adv_ce_losses,
                    factor=factor, kappa = kappa))
    
                    
    logger.log(  '[FINAL RESULT epoch:{:2d} eps:{:.4f}]: '
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
        'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
        'Unet Loss {unetloss.val:.4f} ({unetloss.avg:.4f})  '
        'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
        'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
        'ACE {adv_ce_loss.val:.4f} ({adv_ce_loss.avg:.4f})  '
        'Err {errors.val:.4f} ({errors.avg:.4f})  '
        'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
        'Adv Err {adv_errors.val:.4f} ({adv_errors.avg:.4f})  '
        'beta {factor:.3f} ({factor:.3f})  '
        'kappa {kappa:.3f} ({kappa:.3f})  \n'.format(
        t, eps, batch_time=batch_time,
        loss=losses,unetloss=unetlosses, errors=errors, robust_errors = robust_errors,
        adv_errors = adv_errors,
        regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses,
        adv_ce_loss = adv_ce_losses,
        kappa = kappa, factor=factor))


    if method == "natural":
        return errors.avg, errors.avg
    else:
        return robust_errors.avg, errors.avg



def main(args):
    config = load_config(args)
    global_train_config = config["training_params"]
    models, model_names = config_modelloader(config)

    converted_models = [BoundSequential.convert(model) for model in models]

    for model, model_id, model_config in zip(converted_models, model_names, config["models"]):
        print("Number of GPUs:", torch.cuda.device_count())
        model = model.cuda()
        # make a copy of global training config, and update per-model config
        train_config = copy.deepcopy(global_train_config)
        if "training_params" in model_config:
            train_config = update_dict(train_config, model_config["training_params"])

        # read training parameters from config file
        epochs = train_config["epochs"]
        lr = train_config["lr"]
        weight_decay = train_config["weight_decay"]
        starting_epsilon = train_config["starting_epsilon"]
        end_epsilon = train_config["epsilon"]
        schedule_length = train_config["schedule_length"]
        schedule_start = train_config["schedule_start"]
        optimizer = train_config["optimizer"]
        method = train_config["method"]
        verbose = train_config["verbose"]
        lr_decay_step = train_config["lr_decay_step"]
        lr_decay_factor = train_config["lr_decay_factor"]
        # parameters specific to a training method
        method_param = train_config["method_params"]
        norm = float(train_config["norm"])
        train_config["loader_params"]["batch_size"] = train_config["loader_params"]["batch_size"]//args.grad_acc_steps
        train_config["loader_params"]["test_batch_size"] = train_config["loader_params"]["test_batch_size"]//args.grad_acc_steps
        train_data, test_data = config_dataloader(config, **train_config["loader_params"])

        # initialize adversary network
        if method_param["attack_type"] == "patch-nn":
            if config["dataset"] == "mnist":
                adv_net = ResNetUNet(n_class=10, channels=1,
                                     base_width=method_param["base_width"],
                                     dataset="mnist").cuda()
            if config["dataset"] == "cifar":
                adv_net = ResNetUNet(n_class=10, channels=3,
                                     base_width=method_param["base_width"],
                                     dataset="cifar").cuda()
        else:
            adv_net = None
        if optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            if method_param["attack_type"] == "patch-nn":
                unetopt = optim.Adam(adv_net.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                unetopt = None
        elif optimizer == "sgd":
            if method_param["attack_type"] == "patch-nn":
                unetopt = optim.SGD(adv_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
            else:
                unetopt = None
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer")
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_factor)
        if method_param["attack_type"] == "patch-nn":
            lr_scheduler_unet = optim.lr_scheduler.StepLR(unetopt, step_size=lr_decay_step, gamma=lr_decay_factor)


        start_epoch = 0
        if args.resume:
            model_log = os.path.join(out_path, "test_log")
            logger = Logger(open(model_log, "w"))
            state_dict = torch.load(args.resume)
            print("***** Loading state dict from {} @ epoch {}".format(args.resume, state_dict['epoch']))
            model.load_state_dict(state_dict['state_dict'])
            opt.load_state_dict(state_dict['opt_state_dict'])
            lr_scheduler.load_state_dict(state_dict['lr_scheduler_dict'])
            start_epoch = state_dict['epoch'] + 1

        eps_schedule = [0] * schedule_start + list(np.linspace(starting_epsilon, end_epsilon, schedule_length))
        max_eps = end_epsilon

        model_name = get_path(config, model_id, "model", load = False)
        best_model_name = get_path(config, model_id, "best_model", load = False)
        print(model_name)
        model_log = get_path(config, model_id, "train_log")
        logger = Logger(open(model_log, "w"))
        logger.log("Command line:", " ".join(sys.argv[:]))
        logger.log("training configurations:", train_config)
        logger.log("Model structure:")
        logger.log(str(model))
        logger.log("data std:", train_data.std)
        best_err = np.inf
        recorded_clean_err = np.inf
        timer = 0.0

        for t in range(start_epoch, epochs):
            if method_param["attack_type"] == "patch-nn":
                lr_scheduler_unet.step(epoch=max(t-len(eps_schedule), 0))
            lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))

            if t >= len(eps_schedule):
                eps = end_epsilon
            else:
                epoch_start_eps = eps_schedule[t]
                if t + 1 >= len(eps_schedule):
                    epoch_end_eps = epoch_start_eps
                else:
                    epoch_end_eps = eps_schedule[t+1]
            
            logger.log("Epoch {}, learning rate {}, epsilon {:.6f} - {:.6f}".format(t, lr_scheduler.get_lr(), epoch_start_eps, epoch_end_eps))
            # with torch.autograd.detect_anomaly():
            start_time = time.time()


            Train(model, model_id, t, train_data, epoch_start_eps, epoch_end_eps, max_eps, norm, logger, verbose, True, opt, method, adv_net, unetopt, **method_param)
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.log('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            logger.log("Evaluating...")
            # evaluate
            err, clean_err = Train(model, model_id, t, test_data, epoch_end_eps, epoch_end_eps, max_eps, norm, logger, verbose, False, None, method, adv_net, None, **method_param)


            logger.log('saving to', model_name)
            torch.save({
                    'state_dict' : model.state_dict(),
                    'opt_state_dict': opt.state_dict(),
                    'robust_err': err,
                    'clean_err': clean_err,
                    'epoch' : t,
                    'lr_scheduler_dict': lr_scheduler.state_dict()
                    }, model_name)

            # save the best model after we reached the schedule
            if t >= len(eps_schedule):
                if err <= best_err:
                    best_err = err
                    recorded_clean_err = clean_err
                    logger.log('Saving best model {} with error {}'.format(best_model_name, best_err))
                    torch.save({
                            'state_dict' : model.state_dict(),
                            'opt_state_dict': opt.state_dict(),
                            'robust_err': err,
                            'clean_err': clean_err,
                            'epoch' : t,
                            'lr_scheduler_dict': lr_scheduler.state_dict()
                            }, best_model_name)

        logger.log('Total Time: {:.4f}'.format(timer))
        logger.log('Model {} best err {}, clean err {}'.format(model_id, best_err, recorded_clean_err))


if __name__ == "__main__":
    args = argparser()
    main(args)
