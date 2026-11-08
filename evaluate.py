"""Test kümesinde tüm yöntemlerin değerlendirilmesi ve karşılaştırılması.

Eğitilmiş modelleri (checkpoints/*_best.pt) ve Otsu yöntemini aynı test
hastaları üzerinde değerlendirir. Çıktılar:
    results/metrikler_ozet.csv      -> yöntem başına özet metrikler
    results/metrikler_kesit.csv     -> her kesit için metrikler
    results/sonuclar.md             -> rapora kopyalanabilir tablo
    results/figures/*.png           -> karşılaştırma şekilleri
    demo_ornekler/*.png             -> canlı demo için örnek görüntüler

Kullanım:
    python evaluate.py
    python evaluate.py --threshold 0.5
"""
import argparse
import shutil
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.data_utils import imwrite_any, load_or_create_split, read_image, read_mask
from src.metrics import slice_metrics
from src.otsu_baseline import otsu_segment
from src.visualize import overlay


# --------------------------------------------------------------------------
# Tahmin
# --------------------------------------------------------------------------
def postprocess(pred: np.ndarray) -> np.ndarray:
    """Çok küçük tahminleri (gürültü) sil; tüm yöntemlere aynı kural uygulanır."""
    if pred.sum() < config.MIN_TUMOR_PIXELS:
        return np.zeros_like(pred)
    return pred


def predict_model(ckpt_path, df, device, batch_size, threshold, workers):
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import BrainMRIDataset
    from src.models import count_parameters, load_checkpoint

    model, ckpt = load_checkpoint(ckpt_path, device)
    ds = BrainMRIDataset(df, ckpt["img_size"], config.MEAN, config.STD, train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    probs = []
    elapsed = 0.0
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            p = torch.sigmoid(model(x))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            probs.append(p[:, 0].cpu().numpy().astype(np.float16))
    probs = np.concatenate(probs)
    preds = []
    for prob, (_, row) in zip(probs, df.iterrows()):
        gt_shape = read_mask(row.mask_path).shape
        if prob.shape != gt_shape:  # model farklı boyutta eğitildiyse geri ölçekle
            prob = cv2.resize(prob.astype(np.float32), gt_shape[::-1])
        preds.append(postprocess((prob > threshold).astype(np.uint8)))
    info = {"ms_per_slice": 1000 * elapsed / len(df), "n_params": count_parameters(model),
            "epoch": ckpt.get("epoch")}
    return preds, info


def predict_otsu(df):
    preds = []
    t0 = time.perf_counter()
    for _, row in df.iterrows():
        preds.append(postprocess(otsu_segment(read_image(row.image_path))))
    ms = 1000 * (time.perf_counter() - t0) / len(df)
    return preds, {"ms_per_slice": ms, "n_params": 0, "epoch": None}


# --------------------------------------------------------------------------
# Metrik özetleri
# --------------------------------------------------------------------------
def summarize(per_slice: pd.DataFrame, info: dict) -> pd.DataFrame:
    rows = []
    for method, d in per_slice.groupby("method", sort=False):
        tum = d[d.gt_area > 0]
        emp = d[d.gt_area == 0]
        # Hasta (3B hacim) düzeyinde Dice: hastanın tüm kesitleri birleştirilir.
        g = d.groupby("patient")[["tp", "fp", "fn"]].sum()
        g = g[(g.tp + g.fn) > 0]
        pdice = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
        TP, FP, FN = d.tp.sum(), d.fp.sum(), d.fn.sum()
        det_pred = d.pred_area >= config.MIN_TUMOR_PIXELS
        rows.append({
            "method": method,
            "yontem": config.MODEL_LABELS.get(method, method),
            "dice_tumorlu_kesit": tum.dice.mean(),
            "dice_tumorlu_kesit_std": tum.dice.std(),
            "iou_tumorlu_kesit": tum.iou.mean(),
            "dice_tum_kesitler": d.dice.mean(),
            "dice_hasta_bazli": pdice.mean(),
            "dice_hasta_bazli_std": pdice.std(),
            "dice_global": 2 * TP / (2 * TP + FP + FN),
            "kesinlik_global": TP / (TP + FP) if TP + FP > 0 else np.nan,
            "duyarlilik_global": TP / (TP + FN) if TP + FN > 0 else np.nan,
            "hd95_medyan_piksel": d.hd95.median(),
            "tespit_duyarlilik": det_pred[d.gt_area > 0].mean(),
            "tespit_ozgulluk": (~det_pred[d.gt_area == 0]).mean() if len(emp) else np.nan,
            "ms_kesit": info[method]["ms_per_slice"],
            "parametre_milyon": info[method]["n_params"] / 1e6,
        })
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, path):
    lines = [
        "| Yöntem | Dice (tümörlü kesit) | IoU (tümörlü kesit) | Dice (hasta bazlı) | HD95 medyan (px) "
        "| Tespit duyarlılık | Tespit özgüllük | Süre (ms/kesit) | Parametre (M) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.yontem} | {r.dice_tumorlu_kesit:.3f} ± {r.dice_tumorlu_kesit_std:.3f} "
            f"| {r.iou_tumorlu_kesit:.3f} | {r.dice_hasta_bazli:.3f} ± {r.dice_hasta_bazli_std:.3f} "
            f"| {r.hd95_medyan_piksel:.1f} | {r.tespit_duyarlilik:.3f} | {r.tespit_ozgulluk:.3f} "
            f"| {r.ms_kesit:.1f} | {r.parametre_milyon:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


# --------------------------------------------------------------------------
# Şekiller
# --------------------------------------------------------------------------
def pick_examples(test_df, n):
    """Tümör alanı küçükten büyüğe yayılan, farklı hastalardan n kesit seçer."""
    pos = test_df[test_df.has_tumor].sort_values("tumor_area")
    picked, used = [], set()
    for q in np.linspace(0.15, 0.95, min(n, len(pos))):
        start = min(int(q * len(pos)), len(pos) - 1)
        rest = pos[~pos.index.isin(picked)]
        cand = rest[rest.tumor_area >= pos.tumor_area.iloc[start]]
        if len(cand) == 0:
            cand = rest
        fresh = cand[~cand.patient.isin(used)]  # mümkünse farklı hastadan seç
        row = (fresh if len(fresh) else cand).iloc[0]
        picked.append(row.name)
        used.add(row.patient)
    return picked


def fig_examples(test_df, preds, methods, per_slice, out):
    idx = pick_examples(test_df, 5)
    cols = 2 + len(methods)
    fig, axes = plt.subplots(len(idx), cols, figsize=(2.6 * cols, 2.7 * len(idx)))
    axes = np.atleast_2d(axes)
    pos_of = {k: i for i, k in enumerate(test_df.index)}
    for r, k in enumerate(idx):
        row = test_df.loc[k]
        img, gt = read_image(row.image_path), read_mask(row.mask_path)
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(overlay(img, gt=gt))
        if r == 0:
            axes[r, 0].set_title("MR görüntüsü", fontsize=10)
            axes[r, 1].set_title("Uzman maskesi", fontsize=10)
        for c, m in enumerate(methods):
            p = preds[m][pos_of[k]]
            dsc = per_slice[(per_slice.method == m)].iloc[pos_of[k]].dice
            axes[r, 2 + c].imshow(overlay(img, pred=p, gt=gt))
            axes[r, 2 + c].set_xlabel(f"Dice = {dsc:.2f}", fontsize=9)
            if r == 0:
                axes[r, 2 + c].set_title(config.MODEL_LABELS[m], fontsize=10)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
    fig.text(0.5, 0.005, "Yeşil kontur: uzman maskesi   |   Kırmızı: yöntemin tahmini",
             ha="center", fontsize=10)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(out, dpi=150)
    plt.close()


def fig_distribution(per_slice, methods, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels = [config.MODEL_LABELS[m].replace(" (", "\n(") for m in methods]
    data = [per_slice[(per_slice.method == m) & (per_slice.gt_area > 0)].dice for m in methods]
    axes[0].boxplot(data, showmeans=True)
    axes[0].set_xticks(range(1, len(methods) + 1), labels, fontsize=9)
    axes[0].set_title("Kesit bazlı Dice (tümörlü kesitler)")
    axes[0].set_ylabel("Dice")

    pdata = []
    for m in methods:
        g = per_slice[per_slice.method == m].groupby("patient")[["tp", "fp", "fn"]].sum()
        g = g[(g.tp + g.fn) > 0]
        pdata.append(2 * g.tp / (2 * g.tp + g.fp + g.fn))
    axes[1].boxplot(pdata, showmeans=True)
    axes[1].set_xticks(range(1, len(methods) + 1), labels, fontsize=9)
    axes[1].set_title("Hasta bazlı Dice (3B hacim)")
    axes[1].set_ylabel("Dice")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig_bars(summary, out):
    metrics = [("dice_tumorlu_kesit", "Dice\n(tümörlü kesit)"), ("iou_tumorlu_kesit", "IoU\n(tümörlü kesit)"),
               ("dice_hasta_bazli", "Dice\n(hasta bazlı)"), ("tespit_ozgulluk", "Tespit\nözgüllüğü")]
    x = np.arange(len(metrics))
    w = 0.8 / len(summary)
    colors = ["#95a5a6", "#2980b9", "#8e44ad", "#c0392b"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (_, r) in enumerate(summary.iterrows()):
        vals = [r[k] for k, _ in metrics]
        b = ax.bar(x + (i - (len(summary) - 1) / 2) * w, vals, w, label=r.yontem,
                   color=colors[i % len(colors)])
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [l for _, l in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Skor")
    ax.set_title("Yöntemlerin test kümesinde karşılaştırılması")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig_failures(test_df, preds, method, per_slice, out, n=4):
    d = per_slice[(per_slice.method == method)].reset_index(drop=True)
    worst = d[d.gt_area > 0].sort_values("dice").head(n).index
    fig, axes = plt.subplots(1, len(worst), figsize=(3.2 * len(worst), 3.5))
    axes = np.atleast_1d(axes)
    for a, i in zip(axes, worst):
        row = test_df.iloc[i]
        img, gt = read_image(row.image_path), read_mask(row.mask_path)
        a.imshow(overlay(img, pred=preds[method][i], gt=gt))
        a.set_title(f"Dice = {d.loc[i, 'dice']:.2f}\nalan: {int(d.loc[i, 'gt_area'])} px", fontsize=9)
        a.axis("off")
    fig.suptitle(f"En zor örnekler: {config.MODEL_LABELS[method]}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def export_demo_examples(test_df, n_pos=8, n_neg=2):
    """Canlı demo için test kümesinden PNG örnekler çıkarır (modelin hiç görmediği hastalar)."""
    if config.DEMO_DIR.exists():
        shutil.rmtree(config.DEMO_DIR)
    config.DEMO_DIR.mkdir(parents=True)
    idx = pick_examples(test_df, n_pos)
    neg = test_df[~test_df.has_tumor]
    idx += list(neg.sample(min(n_neg, len(neg)), random_state=config.SEED).index)
    for i, k in enumerate(idx):
        row = test_df.loc[k]
        tag = "tumorlu" if row.has_tumor else "tumorsuz"
        base = f"{i + 1:02d}_{tag}_{row.patient}_{row.slice}"
        img = read_image(row.image_path)
        imwrite_any(config.DEMO_DIR / f"{base}_goruntu.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        imwrite_any(config.DEMO_DIR / f"{base}_maske.png", read_mask(row.mask_path) * 255)
    print(f"Demo örnekleri: {config.DEMO_DIR} ({len(idx)} görüntü)")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_create_split(args.data_dir, config.SPLIT_FILE, config.VAL_RATIO,
                              config.TEST_RATIO, config.SEED)
    test_df = df[df.split == "test"].reset_index(drop=True)
    print(f"Test: {len(test_df)} kesit, {test_df.patient.nunique()} hasta, "
          f"{int(test_df.has_tumor.sum())} tümörlü kesit")

    preds, info = {}, {}
    print("Otsu eşikleme çalışıyor...")
    preds["otsu"], info["otsu"] = predict_otsu(test_df)

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for m in config.MODEL_NAMES:
            ck = config.CHECKPOINT_DIR / f"{m}_best.pt"
            if not ck.exists():
                print(f"  {m}: checkpoint yok, atlanıyor ({ck})")
                continue
            print(f"{config.MODEL_LABELS[m]} değerlendiriliyor...")
            try:
                preds[m], info[m] = predict_model(ck, test_df, device, args.batch_size,
                                                  args.threshold, args.workers)
            except Exception as e:  # bir modelin hatası diğerlerini durdurmasın
                print(f"  HATA ({m}): {e}  -> bu model atlandı")
    except ImportError:
        print("PyTorch bulunamadı; yalnızca Otsu değerlendirildi.")

    methods = list(preds.keys())
    rows = []
    for m in methods:
        for (_, row), p in zip(test_df.iterrows(), preds[m]):
            met = slice_metrics(p, read_mask(row.mask_path))
            rows.append({"method": m, "patient": row.patient, "slice": row.slice, **met})
    per_slice = pd.DataFrame(rows)
    per_slice.to_csv(config.RESULTS_DIR / "metrikler_kesit.csv", index=False)

    summary = summarize(per_slice, info)
    summary.to_csv(config.RESULTS_DIR / "metrikler_ozet.csv", index=False)
    write_markdown(summary, config.RESULTS_DIR / "sonuclar.md")

    fig_examples(test_df, preds, methods, per_slice, config.FIG_DIR / "ornek_tahminler.png")
    fig_distribution(per_slice, methods, config.FIG_DIR / "dice_dagilimi.png")
    fig_bars(summary, config.FIG_DIR / "metrik_karsilastirma.png")
    dl = [m for m in methods if m != "otsu"]
    best = summary[summary.method.isin(dl)].sort_values("dice_tumorlu_kesit").method.iloc[-1] if dl else "otsu"
    fig_failures(test_df, preds, best, per_slice, config.FIG_DIR / "hata_ornekleri.png")
    export_demo_examples(test_df)
    print(f"\nEn iyi yöntem: {config.MODEL_LABELS[best]}")
    print(f"Sonuçlar: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
