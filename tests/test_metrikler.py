"""Metrik fonksiyonlarının birim testleri.

Çalıştırma:  python -m pytest tests      (veya: python kontrol_et.py)
"""
import numpy as np

from src.metrics import dice, hd95, iou, precision_recall, slice_metrics


def _square(x0, y0, s=20, size=64):
    m = np.zeros((size, size), np.uint8)
    m[y0:y0 + s, x0:x0 + s] = 1
    return m


def test_dice_ayni_maske_1():
    a = _square(10, 10)
    assert dice(a, a) == 1.0 and iou(a, a) == 1.0


def test_dice_ortusmeyen_maske_0():
    assert dice(_square(0, 0), _square(40, 40)) == 0.0


def test_iki_bos_maske_1():
    z = np.zeros((64, 64), np.uint8)
    assert dice(z, z) == 1.0 and iou(z, z) == 1.0


def test_yari_ortusme_degerleri():
    a = _square(10, 10, s=20)  # 400 piksel
    b = _square(20, 10, s=20)  # 10 piksel kaydırılmış -> kesişim 200
    assert abs(dice(a, b) - 0.5) < 1e-9  # 2*200/(400+400)
    assert abs(iou(a, b) - 200 / 600) < 1e-9


def test_dice_iou_iliskisi():
    a, b = _square(5, 5, 25), _square(12, 9, 18)
    d, j = dice(a, b), iou(a, b)
    assert abs(d - 2 * j / (1 + j)) < 1e-9  # Dice = 2·IoU / (1 + IoU)


def test_kesinlik_duyarlilik():
    gt = _square(10, 10, 20)
    pred = _square(10, 10, 10)  # tamamen gt içinde
    p, r = precision_recall(pred, gt)
    assert p == 1.0 and abs(r - 100 / 400) < 1e-9


def test_hd95():
    a = _square(10, 10)
    assert hd95(a, a) == 0.0
    assert abs(hd95(a, _square(13, 10)) - 3.0) < 1e-6  # 3 piksel kaydırma
    assert np.isnan(hd95(a, np.zeros_like(a)))  # boş maske: tanımsız


def test_slice_metrics_anahtarlar():
    m = slice_metrics(_square(10, 10), _square(12, 12))
    for k in ["dice", "iou", "precision", "recall", "hd95", "tp", "fp", "fn", "pred_area", "gt_area"]:
        assert k in m
    assert m["tp"] + m["fn"] == m["gt_area"] and m["tp"] + m["fp"] == m["pred_area"]
