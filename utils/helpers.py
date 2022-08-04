import torch
from torch.autograd import Variable


def to_numpy(variable: Variable):
    """
    Converts torch Variable to numpy nd array
    :param variable: torch Variable, of any size
    :return: numpy nd array, of same size as variable
    """
    if torch.cuda.is_available():
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()


def to_cuda_variable_long(tensor, device=0):
    """
    Converts tensor to cuda variable
    :param tensor: torch tensor, of any size
    :return: torch Variable, of same size as tensor
    """
    if torch.cuda.is_available():
        return Variable(tensor.long()).cuda(device)
    else:
        return Variable(tensor.long())


def to_cuda_variable(tensor, device=0):
    """
    Converts tensor to cuda variable
    :param tensor: torch tensor, of any size
    :return: torch Variable, of same size as tensor
    """
    if torch.cuda.is_available():
        return Variable(tensor).cuda(device)
    else:
        return Variable(tensor)
