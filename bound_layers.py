## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import torch
import numpy as np
from torch.nn import Sequential, Conv2d, Linear, ReLU
from model_defs import Flatten, model_mlp_any
import torch.nn.functional as F

import logging
torch.backends.cudnn.determinic = True
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BoundFlatten(torch.nn.Module):
    def __init__(self):
        super(BoundFlatten, self).__init__()

    def forward(self, x):
        self.shape = x.size()[1:]
        return x.view(x.size(0), -1)

    def interval_propagate(self, h_U, h_L, eps):
        return h_U.view(h_U.size(0), -1), h_L.view(h_L.size(0), -1)

class BoundLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BoundLinear, self).__init__(in_features, out_features, bias)

    @staticmethod
    def convert(linear_layer):
        l = BoundLinear(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None)
        l.weight.data.copy_(linear_layer.weight.data)
        l.bias.data.copy_(linear_layer.bias.data)
        return l

    def interval_propagate(self, h_U, h_L, eps, C = None, k=None, Sparse = None):
        # merge the specification
        if C is not None:
            # after multiplication with C, we have (batch, output_shape, prev_layer_shape)
            # we have batch dimension here because of each example has different C
            weight = C.matmul(self.weight)
            bias = C.matmul(self.bias)
        else:
            # weight dimension (this_layer_shape, prev_layer_shape)
            weight = self.weight
            bias = self.bias

        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0
        weight_abs = weight.abs()
        if C is not None:
            center = weight.matmul(mid.unsqueeze(-1)) + bias.unsqueeze(-1)
            deviation = weight_abs.matmul(diff.unsqueeze(-1))
            # these have an extra (1,) dimension as the last dimension
            center = center.squeeze(-1)
            deviation = deviation.squeeze(-1)
        elif Sparse is not None:
            # fused multiply-add
            center = torch.addmm(bias, mid, weight.t())
            deviation = torch.sum(torch.topk(weight_abs, k)[0], dim=1) * eps
        else:
            # fused multiply-add
            center = torch.addmm(bias, mid, weight.t())
            deviation = diff.matmul(weight_abs.t())

        upper = center + deviation
        lower = center - deviation
        # output 
        return upper, lower
            


class BoundConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BoundConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    @staticmethod
    def convert(l):
        nl = BoundConv2d(l.in_channels, l.out_channels, l.kernel_size, l.stride, l.padding, l.dilation, l.groups, l.bias is not None)
        nl.weight.data.copy_(l.weight.data)
        nl.bias.data.copy_(l.bias.data)
        logger.debug(nl.bias.size())
        logger.debug(nl.weight.size())
        return nl

    def forward(self, input):
        output = super(BoundConv2d, self).forward(input)
        self.output_shape = output.size()[1:]
        return output

    def interval_propagate(self, h_U, h_L, eps, k=None, Sparse = None):
        if Sparse is not None:
            mid = (h_U + h_L) / 2.0
            weight_sum = torch.sum(self.weight.abs(), 1)
            deviation = torch.sum(torch.topk(weight_sum.view(weight_sum.shape[0], -1), k)[0], dim=1) * eps
            center = F.conv2d(mid, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            ss = center.shape
            deviation = deviation.repeat(ss[2] * ss[3]).view(-1, ss[1]).t().view(ss[1], ss[2], ss[3])
        else:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = self.weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
            center = F.conv2d(mid, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            logger.debug('center %s', center.size())
        upper = center + deviation
        lower = center - deviation
        return upper, lower
    
class BoundReLU(ReLU):
    def __init__(self, prev_layer, inplace=False):
        super(BoundReLU, self).__init__(inplace)
        # ReLU needs the previous layer's bounds
        # self.prev_layer = prev_layer
    
    ## Convert a ReLU layer to BoundReLU layer
    # @param act_layer ReLU layer object
    # @param prev_layer Pre-activation layer, used for get preactivation bounds
    @staticmethod
    def convert(act_layer, prev_layer):
        l = BoundReLU(prev_layer, act_layer.inplace)
        return l

    def interval_propagate(self, h_U, h_L, eps):
        return F.relu(h_U), F.relu(h_L)



class BoundSequential(Sequential):
    def __init__(self, *args):
        super(BoundSequential, self).__init__(*args)

    ## Convert a Pytorch model to a model with bounds
    # @param sequential_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(sequential_model):
        layers = []
        for l in sequential_model:
            if isinstance(l, Linear):
                layers.append(BoundLinear.convert(l))
            if isinstance(l, Conv2d):
                layers.append(BoundConv2d.convert(l))
            if isinstance(l, ReLU):
                layers.append(BoundReLU.convert(l, layers[-1]))
            if isinstance(l, Flatten):
                layers.append(BoundFlatten())
        return BoundSequential(*layers)

    def interval_range(self, x_U=None, x_L=None, eps=None, C=None, k=None, Sparse=None):
        h_U = x_U
        h_L = x_L
        for i, module in enumerate(list(self._modules.values())[:-1]):
            if Sparse is not None and k is not None:
                if i == 0 and (isinstance(module, Linear) or isinstance(module, Conv2d)):
                    h_U, h_L = module.interval_propagate(h_U, h_L, eps, k=k, Sparse=Sparse)
                elif i == 1 and isinstance(module, Linear):
                    h_U, h_L = module.interval_propagate(h_U, h_L, eps, k=k, Sparse=Sparse)
                else:
                    h_U, h_L = module.interval_propagate(h_U, h_L, eps)
            else:
                h_U, h_L = module.interval_propagate(h_U, h_L, eps)

        # last layer has C to merge
        h_U, h_L = list(self._modules.values())[-1].interval_propagate(h_U, h_L, eps, C)

        return h_U, h_L

    def interval_range_pool(self, x_U=None, x_L=None, eps=None, C=None, neighbor=None, pos_patch_width=None, pos_patch_length=None):
        h_U = x_U
        h_L = x_L
        last_module = list(self._modules.values())[-1]

        for i, module in enumerate(list(self._modules.values())[0:-1]):
            h_U, h_L = module.interval_propagate(h_U, h_L, eps)

            #pool bounds
            if i < len(neighbor) and neighbor[i] > 1:
                ori_shape = h_U.shape
                batch_size = ori_shape[0] // pos_patch_width // pos_patch_length
                # h_U = (batch*possible patch, width_bound, length_bound, channels_bound)
                h_U = h_U.view(batch_size, pos_patch_width, pos_patch_length, -1)
                # h_U = (batch, width, length, width_bound*length_bound*channels_bound)
                h_U = h_U.permute(0, 3, 1, 2)
                # h_U = (batch, width_bound*length_bound*channels_bound, width, length)
                h_U = torch.nn.functional.max_pool2d(h_U, neighbor[i], neighbor[i], 0, 1, True, False)
                # h_U = (batch, width_bound*length_bound*channels_bound, (width-1)//neighbor+1, (length-1)//neighbor+1)
                h_U = h_U.permute(0, 2, 3, 1)
                # h_U = (batch, (width-1)//neighbor+1, (length-1)//neighbor+1, width_bound*length_bound*channels_bound)
                h_U = h_U.reshape(-1, *ori_shape[1:])
                # h_U = (batch*(width-1)//neighbor+1*(length-1)//neighbor+1, width_bound*length_bound*channels_bound)

                h_L = h_L.view(batch_size, pos_patch_width, pos_patch_length, -1)
                h_L = h_L.permute(0, 3, 1, 2)
                h_L = -torch.nn.functional.max_pool2d(-h_L, neighbor[i], neighbor[i], 0, 1, True, False)
                h_L = h_L.permute(0, 2, 3, 1)
                h_L = h_L.reshape(-1, *ori_shape[1:])

                pos_patch_width = (pos_patch_width-1)//neighbor[i] + 1
                pos_patch_length = (pos_patch_length-1)//neighbor[i] + 1

        # last layer has C to merge
        h_U, h_L= last_module.interval_propagate(h_U, h_L, eps, C)
        return h_U, h_L

class ParallelBound(torch.nn.Module):
    def __init__(self, model):
        super(ParallelBound, self).__init__()
        self.model = model
    def forward(self, x_U, x_L, eps, C):
        ub, lb = self.model.interval_range(x_U=x_U, x_L=x_L, eps=eps, C=C)
        return ub, lb

class ParallelBoundPool(torch.nn.Module):
    def __init__(self, model):
        super(ParallelBoundPool, self).__init__()
        self.model = model
    def forward(self, x_U,
                x_L, eps, C, neighbor, pos_patch_width, pos_patch_length):
        ub, lb = self.model.interval_range_pool(x_U=x_U, x_L=x_L, eps=eps, C=C, neighbor=neighbor,
                                                pos_patch_width = pos_patch_width, pos_patch_length = pos_patch_length)
        return ub, lb