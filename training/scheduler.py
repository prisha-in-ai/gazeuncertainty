from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR


def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """
    Adam optimizer with weight decay.
    A lower lr is used since we're fine-tuning a pretrained ResNet.
    """
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, scheduler_type="cosine", **kwargs):
    """
    Returns a learning rate scheduler.

    Args:
        optimizer      : the optimizer to wrap
        scheduler_type : "cosine" or "onecycle"
        kwargs         : extra args depending on scheduler type

    For "cosine":
        - num_epochs (int): total training epochs (default: 30)

    For "onecycle":
        - max_lr      (float): peak learning rate (default: 1e-3)
        - steps_per_epoch (int): number of batches per epoch
        - num_epochs  (int): total training epochs
    """
    if scheduler_type == "cosine":
        num_epochs = kwargs.get("num_epochs", 30)
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    elif scheduler_type == "onecycle":
        max_lr           = kwargs.get("max_lr", 1e-3)
        steps_per_epoch  = kwargs.get("steps_per_epoch")
        num_epochs       = kwargs.get("num_epochs", 30)
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for onecycle scheduler.")
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,       # 30% warmup
            anneal_strategy="cos",
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose 'cosine' or 'onecycle'.")