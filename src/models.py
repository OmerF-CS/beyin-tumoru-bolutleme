"""Bölütleme modelleri.

- UNet: Ronneberger vd. (2015) mimarisinin sıfırdan PyTorch uygulaması.
- UNet(attention=True): Oktay vd. (2018) dikkat kapılı (attention gate) U-Net.
- resnet34_unet: ImageNet ile önceden eğitilmiş ResNet34 kodlayıcılı U-Net
  (segmentation_models_pytorch kütüphanesi ile).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(3x3 Conv -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Dikkat kapısı: kod çözücüden gelen sinyal (g) ile atlama bağlantısındaki
    özellikleri (x) ağırlıklandırır; ilgisiz bölgeler bastırılır."""

    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_x = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))  # (B,1,H,W) in [0,1]
        return x * alpha


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32, attention=False):
        super().__init__()
        self.attention = attention
        feats = [base * 2**i for i in range(5)]  # 32, 64, 128, 256, 512

        self.encoders = nn.ModuleList()
        ch = in_ch
        for f in feats[:-1]:
            self.encoders.append(DoubleConv(ch, f))
            ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(feats[-2], feats[-1])

        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(feats[:-1]):  # 256, 128, 64, 32
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.gates.append(AttentionGate(f, f, max(f // 2, 1)) if attention else nn.Identity())
            self.decoders.append(DoubleConv(f * 2, f))
        self.head = nn.Conv2d(feats[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, gate, dec, skip in zip(self.ups, self.gates, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.attention:
                skip = gate(x, skip)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)  # logit (sigmoid uygulanmamış)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "unet":
        return UNet(base=32, attention=False)
    if name == "attention_unet":
        return UNet(base=32, attention=True)
    if name == "resnet34_unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "resnet34_unet için 'segmentation-models-pytorch' gerekli: "
                "pip install segmentation-models-pytorch"
            ) from e
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
    raise ValueError(f"Bilinmeyen model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path, device):
    """Kayıtlı modeli yükler; (model, checkpoint sözlüğü) döndürür."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # eski PyTorch sürümleri
        ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
