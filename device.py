import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def move_to_device(obj, device=None):
    """
    Moves a PyTorch model or tensor to the specified device.
    If no device is provided, the function determines the best available device.
    """
    if device is None:
        device = get_device()
    return obj.to(device)
