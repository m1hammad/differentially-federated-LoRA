import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_device():
    if torch.cuda.is_available():
        logging.info("Using CUDA")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logging.info("Using MPS")
        return torch.device("mps")
    else:
        logging.info("Using CPU")
        return torch.device("cpu")
    
def move_to_device(obj, device=None):
    """
    Moves a PyTorch model or tensor to the specified device.
    If no device is provided, the function determines the best available device.
    """
    if device is None:
        device = get_device()
    return obj.to(device)
