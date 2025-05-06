#REFERENCE: Ben's Batched_AL notebook
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
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

    def forward(self, logits, targets):
        """
        Args:
            logits: Model outputs (before sigmoid), shape (batch_size, num_classes)
            targets: Ground truth labels (binary), shape (batch_size, num_classes)
        Returns:
            Focal loss value
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        p_t = probs * targets + (1 - probs) * (1 - targets)  # Get p_t for correct class
        
        focal_weight = (1 - p_t) ** self.gamma  # Compute focal weight
        if self.alpha is not None:
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_weight * focal_weight
        
        loss = focal_weight * bce_loss  # Apply focal weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none' keeps per-sample loss


