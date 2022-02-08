import numpy as np 
import torch

def to_tensor(*args):
    """
    Convert args to tensors
    """
    def convert(var):
        if isinstance(var, np.ndarray):
            return torch.from_numpy(var)
        elif isinstance(var, torch.Tensor):
            return var
        else:
            raise NotImplementedError("Invalid data structure or type")

    converted_vars = map(lambda x: convert(x), args)
    return converted_vars


def to_ndarray(*args):
    """
    Convert args to ndarray
    """
    def convert(var):
        if isinstance(var, torch.Tensor):
            return var.to("cpu").detach().numpy().copy()
        elif isinstance(var, np.ndarray):
            return var
        else:
            raise NotImplementedError("Invalid data structure or type")

    converted_vars = map(lambda x: convert(x), args)
    return converted_vars
