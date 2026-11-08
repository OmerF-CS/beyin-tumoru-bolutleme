"""Model testleri (PyTorch gerekir; yoksa atlanır)."""
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _need_torch():
    if torch is None:
        raise unittest.SkipTest("PyTorch kurulu değil")


def test_unet_cikis_boyutu_ve_parametre_sayisi():
    _need_torch()
    from src.models import build_model, count_parameters
    for name, n_expected in [("unet", 7_763_041), ("attention_unet", 7_851_293)]:
        m = build_model(name).eval()
        assert count_parameters(m) == n_expected, name  # raporda verilen değerler
        with torch.no_grad():
            y = m(torch.randn(2, 3, 128, 128))
        assert y.shape == (2, 1, 128, 128), name


def test_resnet34_unet():
    _need_torch()
    try:
        import segmentation_models_pytorch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("segmentation-models-pytorch kurulu değil")
    from src.models import build_model, count_parameters
    m = build_model("resnet34_unet", pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.randn(1, 3, 128, 128))
    assert y.shape == (1, 1, 128, 128)
    assert 24.0e6 < count_parameters(m) < 24.9e6


def test_dikkat_kapisi_0_1_araliginda():
    _need_torch()
    from src.models import AttentionGate
    g = AttentionGate(8, 8, 4).eval()
    x, s = torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16)
    out = g(s, x)
    assert out.shape == x.shape
    ratio = (out / x.where(x.abs() > 1e-6, torch.ones_like(x)))
    assert float(ratio.min()) >= -1e-5 and float(ratio.max()) <= 1 + 1e-5


def test_kayip_ve_geri_yayilim():
    _need_torch()
    from src.losses import DiceBCELoss
    from src.models import build_model
    m = build_model("unet")
    x = torch.randn(2, 3, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.8).float()
    loss = DiceBCELoss()(m(x), y)
    loss.backward()
    assert torch.isfinite(loss) and 0 < float(loss) < 2
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_kayip_mukemmel_tahminde_kucuk():
    _need_torch()
    from src.losses import DiceBCELoss
    y = torch.zeros(1, 1, 32, 32)
    y[..., 8:24, 8:24] = 1
    perfect = (y * 2 - 1) * 20  # çok emin ve doğru logitler
    assert float(DiceBCELoss()(perfect, y)) < 0.01


def test_checkpoint_kaydet_yukle():
    _need_torch()
    from src.models import build_model, load_checkpoint
    m = build_model("unet").eval()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "unet_best.pt"
        torch.save({"model_name": "unet", "state_dict": m.state_dict(), "img_size": 128, "epoch": 1}, p)
        m2, ck = load_checkpoint(p, torch.device("cpu"))
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            assert torch.allclose(m(x), m2(x), atol=1e-6)
        assert ck["img_size"] == 128
