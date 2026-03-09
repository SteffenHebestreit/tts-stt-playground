import torch
import torch.optim as optim
import logging
from torch.optim.lr_scheduler import ExponentialLR, StepLR

logger = logging.getLogger(__name__)

def get_optimizer(model, config):
    """Get optimizer for training"""
    return optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.01
    )

def get_scheduler(optimizer, config):
    """Get learning rate scheduler"""
    return ExponentialLR(
        optimizer,
        gamma=0.999875
    )

def get_cosine_scheduler(optimizer, config):
    """Alternative cosine annealing scheduler"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.get('epochs', 1000),
        eta_min=config.get('learning_rate', 2e-4) * 0.01
    )

def get_step_scheduler(optimizer, config):
    """Step scheduler that reduces LR at fixed intervals"""
    return StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 200),
        gamma=config.get('lr_gamma', 0.5)
    )

def count_parameters(model):
    """Count the total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('loss', 0.0)

def initialize_weights(model):
    """Initialize model weights"""
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)

def gradient_clipping(model, max_norm=1.0):
    """Apply gradient clipping"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    return device

def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class LossTracker:
    """Track training losses"""
    
    def __init__(self):
        self.losses = {}
        self.history = []
        
    def update(self, losses_dict):
        for key, value in losses_dict.items():
            if key not in self.losses:
                self.losses[key] = []
            self.losses[key].append(value.item() if torch.is_tensor(value) else value)
        
        total_loss = sum(losses_dict.values())
        self.history.append(total_loss.item() if torch.is_tensor(total_loss) else total_loss)
    
    def get_average(self, window=100):
        if not self.history:
            return 0.0
        return sum(self.history[-window:]) / min(len(self.history), window)
    
    def get_loss_dict_average(self, window=100):
        averages = {}
        for key, values in self.losses.items():
            if values:
                averages[key] = sum(values[-window:]) / min(len(values), window)
        return averages
