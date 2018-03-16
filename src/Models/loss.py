import torch
import torch.nn as nn
from torch.autograd import Variable

nllloss = nn.NLLLoss()

def losses_joint(out, labels: torch._TensorBase, time_steps: int):
    """
    Defines loss
    :param out: output from the network
    :param labels: Ground truth labels
    :param time_steps: Length of the program
    :return loss: Sum of categorical losses 
    """
    loss = Variable(torch.zeros(1)).cuda()

    for i in range(time_steps):
        loss += nllloss(out[i], labels[:, i])
    return loss