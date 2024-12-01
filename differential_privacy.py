from opacus import PrivacyEngine

def differential_privacy(model, optimizer, data_loader, noise_multiplier=1.0, max_grad_norm=1.0): # Configures the model, optimizer, and data loader for differential privacy.

        #model (torch.nn.Module): The model to be trained with differential privacy.
        #optimizer (torch.optim.Optimizer): The optimizer used for training.
        #data_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        #noise_multiplier (float): The ratio of the standard deviation of the Gaussian noise to the L2 norm of the gradients.
        # max_grad_norm (float): The maximum norm of the per-sample gradients.

        # Returns: tuple: A tuple containing the privacy-enabled model, optimizer, and data loader.

    privacy_engine = PrivacyEngine()
    model, optimizer, private_data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return model, optimizer, private_data_loader