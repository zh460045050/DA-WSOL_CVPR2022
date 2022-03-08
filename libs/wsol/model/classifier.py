import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

from ..util import normalize_tensor
from ..util import remove_layer
from ..util import replace_layer
from ..util import initialize_weights
import numpy as np
import torch.nn.functional as F
from .basic import *
from torch.autograd import Function
from torch.autograd import Variable

class CAMClassifier(nn.Module):

    def __init__(self, num_classes, num_feature):
        super(CAMClassifier, self).__init__()
        
        self.fc = nn.Conv2d(num_feature, num_classes, 1, 1, padding=0)
        initialize_weights(self.modules(), init_mode='xavier')
    
    def forward(self, x, labels=None, return_cam=False):
        
        batch_size = x.shape[0]
        output = self.fc(x)

        if return_cam:
            normalized = normalize_tensor(output.detach().clone())
            cams = normalized[range(batch_size), labels]
            return cams

        return output



class BCAMClassifier(nn.Module):

    def __init__(self, num_classes):
        super(BCAMClassifier, self).__init__()
        self.extractor_A = nn.Conv2d(512 * block.expansion, num_head, 1, bias=False)
        self.fc = nn.Conv2d(2048, num_classes, 1, 1, padding=0)
        initialize_weights(self.modules(), init_mode='xavier')
    
    def forward(self, feature, labels=None, return_cam=False):
        
        batch_size = x.shape[0]
        att_map_A = self.extractor_A(feature)
        b, c, h, w = att_map_A.shape
        att_map_A = torch.softmax(att_map_A.view(b, c, h*w), dim=-1)
        f_pixel = feature.view(b, -1, h*w)
        x = torch.bmm(att_map_A, f_pixel.permute(0, 2, 1)).mean(dim=1).view(b, -1, 1, 1)
        output = self.fc(x)

        if return_cam:
            output_pixel = self.fc(feature)
            normalized = normalize_tensor(output_pixel.detach().clone())
            cams = normalized[range(batch_size), labels]
            return cams

        return output

class DomainClassifier(nn.Module):

    def __init__(self, num_feature):
        super(DomainClassifier, self).__init__()
        
        self.fc = nn.Conv2d(num_feature, 1, 1, 1, padding=0)
        initialize_weights(self.modules(), init_mode='xavier')
    
    def forward(self, x):
        
        output = self.fc(x)
        output = torch.sigmoid(output)

        return output


class LocalClassifier(nn.Module):

    def __init__(self, num_feature, num_classes):
        super(LocalClassifier, self).__init__()
        
        self.local_classifier = torch.nn.ModuleList()
        for c in range(num_classes):
            self.local_classifier.append( nn.Conv2d(num_feature, 1, 1, 1, padding=0) )
        initialize_weights(self.modules(), init_mode='xavier')
    
    def forward(self, x, target):
        
        #print(x.shape, target.shape)
        x = x.view(target.shape[0], -1, x.shape[1], 1, 1)
        output = Variable(torch.zeros(x.shape[0], x.shape[1], 1, 1, 1)).cuda()
        #print(target.shape, x.shape)
        for i in range(0, x.shape[0]):
            output[i, :, :, :, :] = self.local_classifier[target[i]](x[i, :, :, :, :])
        output = output.view(-1, 1, 1, 1)
        output = torch.sigmoid(output)

        return output


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
