"""Görselleştirme yardımcıları (maske bindirme)."""
import cv2
import numpy as np

PRED_COLOR = (255, 60, 60)  # kırmızı: model tahmini
GT_COLOR = (60, 255, 60)  # yeşil: uzman (gerçek) maskesi


def overlay(img_rgb, pred=None, gt=None, alpha=0.35):
    """Tahmini yarı saydam kırmızı dolgu + kontur, gerçek maskeyi yeşil kontur
    olarak MR görüntüsünün üzerine çizer."""
    out = img_rgb.copy()
    if pred is not None and pred.any():
        color = np.zeros_like(out)
        color[pred.astype(bool)] = PRED_COLOR
        m = pred.astype(bool)
        out[m] = (out[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
        cnts, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, PRED_COLOR, 1)
    if gt is not None and gt.any():
        cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, GT_COLOR, 1)
    return out


def prob_heatmap(img_rgb, prob, alpha=0.5):
    """Tümör olasılık haritasını ısı haritası olarak bindirir."""
    heat = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (img_rgb * (1 - alpha) + heat * alpha).astype(np.uint8)
