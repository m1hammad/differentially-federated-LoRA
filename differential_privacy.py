from opacus import PrivacyEngine
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)

def differential_privacy(model, optimizer, data_loader, noise_multiplier=1.0, max_grad_norm=1.0): # Configures the model, optimizer, and data loader for differential privacy.

    #model (torch.nn.Module): The model to be trained with differential privacy.
    #optimizer (torch.optim.Optimizer): The optimizer used for training.
    #data_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    #noise_multiplier (float): The ratio of the standard deviation of the Gaussian noise to the L2 norm of the gradients.
    # max_grad_norm (float): The maximum norm of the per-sample gradients.

    batch_size = data_loader.batch_size
    sample_size = len(data_loader.dataset)

    logging.info(f"DP module, DataLoader Batch Size: {batch_size}")
    logging.info(f"DP module, DataLoader Sample Size: {sample_size}")
    
    if batch_size == 0 or sample_size == 0:
        raise ValueError("Batch size or dataset size cannot be zero.")
    
    privacy_engine = PrivacyEngine()
    model, optimizer, private_data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    # Log the shapes in the private data loader
    for batch in private_data_loader:
        logging.info(f"DP module, Private batch input_ids shape: {batch['input_ids'].shape}")
        logging.info(f"DP module, Private batch labels shape: {batch['label'].shape}")
        break  # Log one batch for debugging

    return model, optimizer, private_data_loader