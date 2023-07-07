import torch
import torch.nn as nn
import torch.nn.functional as F

__ALL__ = ['FocalLoss_BCE','FocalLoss','DiceLoss']


def idx_2_one_hot(y_idx,nCls):
    """
    y_idx:  LongTensor shape:[1,batch_size] or [batch_size,1]
    y_one_hot:FloatTensor shape:[batch_size, nCls]
    """
    y_idx = y_idx.long().view(-1,1)
    batch_size = y_idx.shape[0]
    y_one_hot = torch.zeros(batch_size,nCls).to(y_idx.device)
    y_one_hot.scatter_(1, y_idx, 1.)
    return y_one_hot

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins: Self-Supervised Learning via Redundancy Reduction[2021][facebook]
    ref: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, L, scale_loss = 1, lambd = 0.1):
        super(BarlowTwinsLoss,self).__init__()
        self.L = L
        self.scale_loss = scale_loss
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(L, affine=False)#without learnable parameter

    def forward(self, x1, x2 ):
        """
        :param x1: [B, L] 
        :param x2: [B, L]
        :return: 
        """
        B, L = x1.shape
        x1 = self.bn(x1)
        x2 = self.bn(x2)
        c = x1.T @ x2 / B / L #[L,L]
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = on_diag + self.lambd * off_diag
        return loss



class DiceLoss(nn.Module):
    def __init__(self, num_class, alpha=None):
        super(DiceLoss,self).__init__()
        self.num_class = num_class
        if isinstance(alpha,float):
            self.alpha = torch.FloatTensor([alpha,1-alpha])
        elif isinstance(alpha,list):
            self.alpha = torch.FloatTensor(alpha)#[ncls]
        else:
            self.alpha = alpha


    def forward(self, input, label):
        # input = [B,ncls]
        input = F.softmax(input,dim=1)
        label = idx_2_one_hot(label,self.num_class)#[B,ncls]
        intersection = torch.sum(input*label,dim=0) #[ncls]
        # print(intersection.shape)
        E_input = input.sum(dim=0)
        E_label = label.sum(dim=0)
        loss = 1 - ( 2*intersection + 1 ) / ( E_input + E_label + 1 )
        if self.alpha is not None:
            self.alpha = self.alpha.to(device=input.device)
            loss*= self.alpha #[ncls]
        loss = loss.sum()
        return loss

class FocalLoss_BCE(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss_BCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        :param input: 
        :param target: [B,1] or [B] discrete long value 
        :return: 
        """
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        # pt = Variable(logpt.data.exp())
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()