import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class TabularFocalLoss(nn.Module):
    """
    Focal Loss focada em dados desbalanceados.
    Reduz o peso de gradientes para exemplos 'fáceis' e foca nos 'hard negatives'.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Calculamos a Cross-Entropy clássica primeiro
        # Usamos F.cross_entropy porque ela embute a LogSoftmax, garantindo estabilidade numérica extrema.
        # reduction='none' nos devolve o erro individual de cada linha do lote.
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 2. Recuperamos a probabilidade pt exata da classe correta usando a inversa do log
        # Como ce_loss = -log(pt), então pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 3. Aplicamos a matemática da Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 4. Redução final
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss