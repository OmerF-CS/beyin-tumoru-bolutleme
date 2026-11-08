"""Kayıp fonksiyonları.

Tümör pikselleri bir kesitin genellikle küçük bir kısmını oluşturur (sınıf
dengesizliği). Yalnızca BCE kullanılırsa model "her yer arka plan" demeye
eğilimli olur. Dice kaybı örtüşmeyi doğrudan optimize ettiği için bu sorunu
hafifletir; BCE ise kararlı, piksel düzeyinde gradyan sağlar.
"""
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        union = prob.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss
