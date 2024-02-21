import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from inspect import isfunction

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        # self.mse_loss = nn.MSELoss(reduce = False)
        self.mae_loss = nn.L1Loss(reduction='none')  # reduce = False

    def forward(self, out_labels, out_images, target_images, epoch):
        # Adversarial Loss
        adversarial_loss = -torch.mean(out_labels)
        # adversarial_loss.requires_grad = True

        # Perception Loss
        # perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        # Image Loss
        image_loss = self.mae_loss(out_images, target_images)
        # image_loss.requires_grad = True

        combined_loss = image_loss.mean() + 0.01 * adversarial_loss / (epoch + 1)  # + 0.006 * perception_loss
        return combined_loss


def mse_loss(output, target):
    return F.mse_loss(output, target)


def mae_loss(output, target):
    return F.l1_loss(output, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()