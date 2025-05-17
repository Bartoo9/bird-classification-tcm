#REFERENCE: Ben's Batched_AL notebook, only added pos_weight in order to use it with the BCEWithLogitsLoss
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', pos_weight=None):
        """
        Args:
            gamma: focusing parameter (default: 2.0)
            alpha: weight for positive class (default: None, meaning equal weight)
            reduction: 'mean', 'sum', or 'none' (like BCEWithLogitsLoss)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be set to balance positive and negative samples
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model outputs (before sigmoid), shape (batch_size, num_classes)
            targets: Ground truth labels (binary), shape (batch_size, num_classes)
        Returns:
            Focal loss value
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        pt = torch.where(targets == 1, probs, 1-probs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1-self.alpha)
            focal_weight = alpha_weight * focal_weight
        
        loss = focal_weight * bce_loss
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


