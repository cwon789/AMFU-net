import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def DiceLoss(inputs, targets, smooth=1):
        
    inputs = torch.sigmoid(inputs) 
        
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
    return 1 - dice 



def BCELoss(pred, target):

    loss_func = nn.BCEWithLogitsLoss()
    loss = loss_func(pred, target)

    return loss



def FocalIoULoss(inputs, targets):
    "Non weighted version of Focal Loss"

    [b,c,h,w] = inputs.size()
    inputs = 0.999*(inputs-0.5)+0.5
    BCE_loss = BCELoss(inputs, targets)
    intersection = torch.mul(inputs, targets)
    smooth = 1

    IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)
    alpha = 0.75
    gamma = 2
    num_classes = 2
    gamma = gamma
    size_average = True
    pt = torch.exp(-BCE_loss)
    F_loss =  torch.mul(((1-pt) ** gamma) ,BCE_loss)
    at = targets*alpha+(1-targets)*(1-alpha)
    F_loss = (1-IoU)*(F_loss)**(IoU*0.5+0.5)
    F_loss_map = at * F_loss
    F_loss_sum = F_loss_map.sum()   # Wrong loss function ? What is it?

    return F_loss_sum



def SoftIoULoss( pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss



def FocalLoss(inputs, targets):
    "Non weighted version of Focal Loss"

    alpha = 0.75
    gamma = 2
    num_classes = 2
    gamma = gamma
    size_average = True
    BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
    targets = targets.type(torch.long)
    at = targets*alpha+(1-targets)*(1-alpha)
    pt = torch.exp(-BCE_loss)
    F_loss = (1 - pt) ** gamma * BCE_loss
    F_loss = at * F_loss

    return F_loss.sum()



def FocalLoss_2(inputs, targets):

    loss_func = FocalLoss2()
    F_loss = loss_func(inputs, targets)

    return F_loss


class FocalLoss2(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



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