import torch
import numpy as np
from torch.autograd import Variable

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    ####
    if source.size()[0] != target.size()[0]:
        if source.size()[0] < target.size()[0]:
            source = source.unsqueeze(0)
            source = source.expand( ( np.int64(target.size()[0] / source.size()[1]), source.size()[1], source.size()[2]))
            source = source.contiguous().view(target.size())
        else:
            target = target.unsqueeze(0)
            target = target.expand( (np.int64(source.size()[0] / target.size()[1]), target.size()[1], target.size()[2]))
            target = target.contiguous().view(source.size())
    ####
    
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


def cal_mmd(batch_source, batch_target, count):

    loss_mmd = Variable(torch.zeros(1)).cuda()
    for i in range(0, count):
        #if self.sample_num_target == self.sample_num_source:
        loss_mmd += mmd_rbf_accelerate(batch_source[i, :, :], batch_target[i, :, :])
        #else:
        #    loss_mud += mmd_rbf_noaccelerate(batch_source[i, :, :], batch_target[i, :, :])
    loss_mmd /= count

    return loss_mmd