import torch
import numpy as np
import pdb

class PGDAttacker:
    def __init__(self, model, mean, std, eps, kwargs):
        std = torch.tensor(std)
        mean = torch.tensor(mean)
        #self.epsilon = kwargs['epsilon'] / std
        self.epsilon = eps / std
        self.steps = kwargs["steps"]
        self.step_size = kwargs["step_size"] / std
        self.model = model
        self.mean = mean
        self.std = std
        self.random_start = kwargs["random_start"]

        self.lb = - mean / std
        self.lb.to('cuda')
        self.ub = (1 - mean) / std
        self.ub.to('cuda')
        #self.sp_lvl = kwargs["sparse_level"]

        self.criterion = torch.nn.CrossEntropyLoss()

    def perturb(self, inputs, labels, norm, ball=None):
        if self.random_start:
            x = inputs + torch.Tensor(np.random.uniform(-self.epsilon, self.epsilon, inputs.size())).to('cuda')
            x = torch.clamp(x, self.lb, self.ub).detach()  # ensure valid pixel range
        else:
            x = inputs.data.detach().clone()

        x_init = inputs.data.detach().clone()
        worst_x = x_init.data.detach()
        #with torch.no_grad():
        #    output = self.model(x)
        #    worst_loss = self.criterion(output, labels)

        x.requires_grad_()
        step = 0
        ss = x.shape
        alterable_pixels = torch.ones_like(x).view(ss[0], ss[1], -1)
        ones = torch.ones_like(x)

        for step in range(self.steps):
            output = self.model(x)
            loss = self.criterion(output, labels)
            if step == 0:
                worst_loss = loss
            grads = torch.autograd.grad(loss, [x])[0]

            if norm == float('inf'):
                signed_grad_x = torch.sign(grads)
                delta = signed_grad_x * self.step_size[None, :, None, None].cuda()
                x.data = delta + x.data.detach()
            elif norm == float('2'):
                delta = grads * self.step_size / grads.view(x.shape[0], -1).norm(2, dim=-1).view(-1, 1, 1, 1)
                x.data = delta + x.data.detach()
            elif norm == float('1'):
                ch = x_init.shape[1]
                ## change mean, std shape (ch, im, im)
                meant = self.mean.repeat(ss[2]*ss[3],1).t().float().cuda()
                stdt = self.std.repeat(ss[2]*ss[3],1).t().float().cuda()

                ## max value can change
                m = (((ones+torch.sign(grads))/2 - meant.view(-1,ss[2],ss[3]))/stdt.view(-1,ss[2],ss[3]) - x)
                grads[m == 0] = 0
                grads_abs = torch.abs(grads)
                batch_size = grads.shape[0]
                if ch == 1:
                    view = grads_abs.view(batch_size, -1)
                    view_size = view.shape[1]
                    sl = 0.99#((0.99 - 0.85)*torch.rand(1) + 0.85).cpu().numpy()[0]
                    vals, idx = view.topk(int(np.round((1 - sl) * view_size)))
                    #vals, idx = view.topk(1)
                    out = torch.zeros_like(view).scatter_(1, idx, vals)
                    out = out.view_as(grads)
                    g = torch.sign(grads) * (out > 0).float()
                    g = g/g.view(batch_size, -1).norm(1, dim=-1).view(-1, 1, 1, 1)
                    delta = g * self.step_size[None, :, None, None].cuda()
                    x.data = delta + x.data.detach()
                else:
                    view = grads_abs.sum(1).view(batch_size, -1)
                    view_size = view.shape[1]
                    sl = ((0.99 - 0.85)*torch.rand(1) + 0.85).cpu().numpy()[0]
                    vals, idx = view.topk(int(np.round((1 - sl) * view_size)))
                    #vals, idx = view.topk(1)
                    out = torch.zeros_like(view).scatter_(1, idx, vals)
                    out = out.repeat(1,ch).view_as(grads)
                    #pdb.set_trace()
                    g = torch.sign(grads) * (out > 0).float()
                    g = g/g.view(batch_size, -1).norm(1, dim=-1).view(-1, 1, 1, 1)
                    delta = g * self.step_size[None, :, None, None].cuda()
                    x.data = delta + x.data.detach()
                #pdb.set_trace()
                #delta = torch.min(torch.max(x_init + delta, self.lb[None, :, None, None].cuda()), self.ub[None, :, None, None].cuda()) - x_init
                 
            elif norm == float('0'):
                #with torch.no_grad():
                    ## change mean, std shape (ch, im, im)
                    #meant = self.mean.repeat(ss[2]*ss[3],1).t().float().cuda()
                    #stdt = self.std.repeat(ss[2]*ss[3],1).t().float().cuda()

                    ## max value can change
                    #m = (((ones+torch.sign(grads))/2 - meant.view(-1,ss[2],ss[3]))/stdt.view(-1,ss[2],ss[3]) - x) * grads

                    ## each pixel can only change once
                    #if step > 1:
                    #    alterable_pixels[torch.arange(ss[0]), :, argmax_] = 0.0
                    #    m = m * alterable_pixels.view(ss[0], ss[1], ss[2], ss[3])

                    ## consider ch together if multi-ch
                    #msum = m.sum(1).view(ss[0], -1)

                    #if msum.sum() == 0:
                    #    break

                    ## argmax_ for each in a batch, return size == batch_size
                    #argmax_ = msum.argmax(-1)

                    ## change selected pixel into lb or ub
                    #x.view(ss[0], ss[1], -1)[torch.arange(ss[0]), :, argmax_] = ((torch.sign(grads.view(ss[0], ss[1], -1)[torch.arange(ss[0]), :, argmax_])+1)/2 - self.mean.cuda()) / self.std.cuda()

                    #x.data = torch.min(torch.max(x, self.lb[None, :, None, None].cuda()), self.ub[None, :, None, None].cuda())

                ## max value can change
                m = (((ones+torch.sign(grads))/2 - meant.view(-1,ss[2],ss[3]))/stdt.view(-1,ss[2],ss[3]) - x)
                grads[m == 0] = 0
                grads_abs = torch.abs(grads)
                batch_size = grads.shape[0]
                if ch == 1:
                    view = grads_abs.view(batch_size, -1)
                    view_size = view.shape[1]
                    sl = 0.99#((0.99 - 0.85)*torch.rand(1) + 0.85).cpu().numpy()[0]
                    vals, idx = view.topk(int(np.round((1 - sl) * view_size)))
                    #vals, idx = view.topk(1)
                    out = torch.zeros_like(view).scatter_(1, idx, vals)
                    out = out.view_as(grads)
                    g = torch.sign(grads) * (out > 0).float()
                    g = g/g.view(batch_size, -1).norm(1, dim=-1).view(-1, 1, 1, 1)
                    delta = g * self.step_size[None, :, None, None].cuda()
                    x.data = delta + x.data.detach()
                else:
                    view = grads_abs.sum(1).view(batch_size, -1)
                    view_size = view.shape[1]
                    sl = ((0.99 - 0.85)*torch.rand(1) + 0.85).cpu().numpy()[0]
                    vals, idx = view.topk(int(np.round((1 - sl) * view_size)))
                    #vals, idx = view.topk(1)
                    out = torch.zeros_like(view).scatter_(1, idx, vals)
                    out = out.repeat(1,ch).view_as(grads)
                    #pdb.set_trace()
                    g = torch.sign(grads) * (out > 0).float()
                    g = g/g.view(batch_size, -1).norm(1, dim=-1).view(-1, 1, 1, 1)
                    delta = g * self.step_size[None, :, None, None].cuda()
                    x.data = delta + x.data.detach()


            # Project back into constraints ball and correct range
            x.data = self.project(x_init, x.data, norm, self.epsilon)
            x.data = torch.min(torch.max(x.data, self.lb[None, :, None, None].cuda()), self.ub[None, :, None, None].cuda())
            
            #with torch.no_grad():
            #    output = self.model(x)
            #    loss = self.criterion(output, labels)
            #    if loss > worst_loss:
            #        worst_loss = loss.detach()
            #        worst_x = x.data.detach()
        return x.data.detach()

    def project(self, x, x_adv, ball, eps):
        if ball == float('inf'):
            x_adv = torch.max(torch.min(x_adv, x + eps[None,:,None,None].cuda()), x - eps[None,:,None,None].cuda())
        elif ball == float('2'):
            # Assume x and x_adv are batched tensors where the first dimension is
            # a batch dimension
            mask = delta.view(delta.shape[0], -1).norm(2, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(2, dim=1)
            scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta
        elif ball == float('1'):
            #eps = eps.sum()
            const = 1e-5
            delta = (x_adv - x).detach().clone()
            batch_size = delta.size(0)
            ch = delta.size(1)
            if ch == 1:
                view = delta.view(batch_size, -1)
                # Computing the l1 norm of v
                v = torch.abs(view)
                v = v.sum(dim=1)
                #pdb.set_trace()
                # Getting the elements to project in the batch
                indexes_b = torch.nonzero(v > (eps.cuda() + const)).view(-1)
                x_b = view[indexes_b]
                batch_size_b = x_b.size(0)

                # If all elements are in the l1-ball, return x
                if batch_size_b == 0:
                    x_adv = delta + x 
                else:
                    # make the projection on l1 ball for elements outside the ball
                    view = x_b
                    view_size = view.size(1)
                    mu = view.abs().sort(1, descending=True)[0]
                    vv = torch.arange(view_size).float().cuda()
                    st = (mu.cumsum(1) - eps.cuda()) / (vv + 1)
                    u = (mu - st) > 0
                    rho = (1 - u).cumsum(dim=1).eq(0).sum(1) - 1
                    theta = st.gather(1, rho.unsqueeze(1))
                    #proj_x_b = _thresh_by_magnitude(theta, x_b)
                    proj_x_b = torch.relu(torch.abs(x_b) - theta) * x_b.sign()

                    # gather all the projected batch
                    proj_x = delta.view(batch_size, -1).detach().clone()
                    proj_x[indexes_b] = proj_x_b
                    x_adv = proj_x.view_as(delta) + x
            else:
                for i in range(ch):
                    view = delta[:,i,:,:].view(batch_size, -1)
                    # Computing the l1 norm of v
                    v = torch.abs(view)
                    v = v.sum(dim=-1)
                    #pdb.set_trace()
                    # Getting the elements to project in the batch
                    indexes_b = torch.nonzero(v > (eps[i].cuda() + const)).view(-1)
                    x_b = view[indexes_b]
                    batch_size_b = x_b.size(0)

                    # If all elements are in the l1-ball, return x
                    if batch_size_b == 0:
                        x_adv[:,i, :, :] = delta[:,i,:,:] + x[:,i,:,:] 
                    else:
                        # make the projection on l1 ball for elements outside the ball
                        view = x_b
                        view_size = view.size(1)
                        mu = view.abs().sort(1, descending=True)[0]
                        vv = torch.arange(view_size).float().cuda()
                        st = (mu.cumsum(1) - eps[i].cuda()) / (vv + 1)
                        u = (mu - st) > 0
                        rho = (1 - u).cumsum(dim=1).eq(0).sum(1) - 1
                        theta = st.gather(1, rho.unsqueeze(1))
                        #proj_x_b = _thresh_by_magnitude(theta, x_b)
                        proj_x_b = torch.relu(torch.abs(x_b) - theta) * x_b.sign()

                        # gather all the projected batch
                        proj_x = delta[:,i,:,:].view(batch_size, -1).detach().clone()
                        proj_x[indexes_b] = proj_x_b
                        x_adv[:,i,:,:] = proj_x.view_as(delta[:,i,:,:]) + x[:,i,:,:]
        elif ball == float('0'):
            delta = (x_adv - x).detach().clone()
            
            x_adv = x + delta

        return x_adv

