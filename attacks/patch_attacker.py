import torch
import numpy as np
import pdb

class PatchAttacker:
    def __init__(self, model, mean, std, kwargs):
        std = torch.tensor(std)
        mean = torch.tensor(mean)
        self.epsilon = kwargs["epsilon"] / std
        self.steps = kwargs["steps"]
        self.step_size = kwargs["step_size"] / std
        self.step_size.cuda()
        self.model = model
        self.mean = mean
        self.std = std
        self.random_start = kwargs["random_start"]

        self.lb = (-mean / std)
        self.lb.to('cuda')
        self.ub = (1 - mean) / std
        self.ub.to('cuda')
        self.patch_w = kwargs["patch_w"]
        self.patch_l = kwargs["patch_l"]

        self.criterion = torch.nn.CrossEntropyLoss()

    def perturb(self, inputs, labels, norm, random_count=1):
        worst_x = None
        worst_loss = None
        for _ in range(random_count):
            # generate random patch center for each image
            idx = torch.arange(inputs.shape[0])[:, None]
            zero_idx = torch.zeros((inputs.shape[0],1), dtype=torch.long)
            w_idx = torch.randint(0, inputs.shape[2]-self.patch_w, (inputs.shape[0],1))
            l_idx = torch.randint(0, inputs.shape[3]-self.patch_l, (inputs.shape[0],1))
            idx = torch.cat([idx,zero_idx, w_idx, l_idx], dim=1)
            idx_list = [idx]
            for w in range(self.patch_w):
                for l in range(self.patch_l):
                    idx_list.append(idx + torch.tensor([0,0,w,l]))
            idx_list = torch.cat(idx_list, dim =0)

            # create mask
            mask = torch.zeros([inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]],
                               dtype=torch.bool).cuda()
            mask[idx_list[:,0],idx_list[:,1],idx_list[:,2],idx_list[:,3]] = True

            if self.random_start:
                init_delta = np.random.uniform(-self.epsilon, self.epsilon,
                                               [inputs.shape[0]*inputs.shape[2]*inputs.shape[3], inputs.shape[1]])
                init_delta = init_delta.reshape(inputs.shape[0],inputs.shape[2],inputs.shape[3], inputs.shape[1])
                init_delta = init_delta.swapaxes(1,3).swapaxes(2,3)
                x = inputs + torch.where(mask, torch.Tensor(init_delta).to('cuda'), torch.tensor(0.).cuda())

                x = torch.min(torch.max(x, self.lb[None, :, None, None].cuda()), self.ub[None, :, None, None].cuda()).detach()  # ensure valid pixel range
            else:
                x = inputs.data.detach().clone()

            x_init = inputs.data.detach().clone()

            x.requires_grad_()

            for step in range(self.steps):
                output = self.model(torch.where(mask, x, x_init))
                loss_ind = torch.nn.CrossEntropyLoss(reduction='none')(output, labels)
                if worst_loss is None:
                    worst_loss = loss_ind.data.detach()
                    worst_x = x.data.detach()
                else:
                    worst_x = torch.where(worst_loss.ge(loss_ind.detach())[:, None, None, None], worst_x, x)
                    worst_loss = torch.where(worst_loss.ge(loss_ind.detach()), worst_loss, loss_ind)
                loss = loss_ind.sum()
                grads = torch.autograd.grad(loss, [x])[0]

                if norm == float('inf'):
                    signed_grad_x = torch.sign(grads).detach()
                    delta = signed_grad_x * self.step_size[None, :, None, None].cuda()
                elif norm == 'l2':
                    delta = grads * self.step_size / grads.view(x.shape[0], -1).norm(2, dim=-1).view(-1, 1, 1, 1)

                x.data = delta + x.data.detach()

                # Project back into constraints ball and correct range
                x.data = self.project(x_init, x.data, norm, self.epsilon)
                x.data = x = torch.min(torch.max(x, self.lb[None, :, None, None].cuda()), self.ub[None, :, None, None].cuda())

        return worst_x

    def project(self, x, x_adv, norm, eps, random=1):
        if norm == 'linf':
            x_adv = torch.max(torch.min(x_adv, x + eps[None, :, None,None]), x - eps[None, :, None,None])
        elif norm == 'l2':
            delta = x_adv - x

            # Assume x and x_adv are batched tensors where the first dimension is
            # a batch dimension
            mask = delta.view(delta.shape[0], -1).norm(2, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(2, dim=1)
            scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta

        return x_adv
