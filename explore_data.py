"""Veri keşfi: istatistikler ve örnek görseller (rapor/sunum için).

Kullanım:
    python explore_data.py
    python explore_data.py --data-dir D:/veriler/kaggle_3m
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.data_utils import load_or_create_split, read_image, read_mask
from src.visualize import overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    args = ap.parse_args()

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_create_split(
        args.data_dir, config.SPLIT_FILE, config.VAL_RATIO, config.TEST_RATIO, config.SEED
    )

    # ------------------------------------------------------------------ istatistik
    pos = df[df.has_tumor]
    sample_img = read_image(df.image_path.iloc[0])
    stats = {
        "hasta_sayisi": int(df.patient.nunique()),
        "kesit_sayisi": int(len(df)),
        "tumorlu_kesit": int(df.has_tumor.sum()),
        "tumorsuz_kesit": int((~df.has_tumor).sum()),
        "tumorlu_oran_yuzde": round(100 * df.has_tumor.mean(), 2),
        "goruntu_boyutu": list(sample_img.shape),
        "tumor_alani_ortalama_piksel": round(float(pos.tumor_area.mean()), 1),
        "tumor_alani_medyan_piksel": float(pos.tumor_area.median()),
        "tumor_piksel_orani_yuzde (tumorlu kesitlerde)": round(
            100 * float(pos.tumor_area.mean()) / (sample_img.shape[0] * sample_img.shape[1]), 2
        ),
        "hasta_basina_kesit_ortalama": round(float(df.groupby("patient").size().mean()), 1),
        "bolme": {},
    }
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        stats["bolme"][s] = {
            "hasta": int(d.patient.nunique()),
            "kesit": int(len(d)),
            "tumorlu_kesit": int(d.has_tumor.sum()),
        }
    with open(config.RESULTS_DIR / "veri_istatistikleri.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------ örnek kesitler
    rng = np.random.default_rng(config.SEED)
    big = pos[pos.tumor_area > pos.tumor_area.quantile(0.4)]
    chosen = big.groupby("patient").sample(1, random_state=config.SEED)
    chosen = chosen.sample(min(4, len(chosen)), random_state=config.SEED)

    titles = ["Kontrast öncesi", "FLAIR", "Kontrast sonrası", "Uzman maskesi (yeşil)"]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(12, 3 * len(chosen)))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(chosen.iterrows()):
        img = read_image(row.image_path)
        mask = read_mask(row.mask_path)
        for c in range(3):
            axes[r, c].imshow(img[..., c], cmap="gray")
        axes[r, 3].imshow(overlay(img, gt=mask))
        for c in range(4):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=12)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "veri_ornekleri.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------ dağılımlar
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    counts = [stats["tumorlu_kesit"], stats["tumorsuz_kesit"]]
    axes[0].bar(["Tümörlü", "Tümörsüz"], counts, color=["#c0392b", "#7f8c8d"])
    for i, v in enumerate(counts):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[0].set_title("Kesit sınıf dağılımı")
    axes[0].set_ylabel("Kesit sayısı")

    axes[1].hist(pos.tumor_area, bins=40, color="#c0392b", alpha=0.85)
    axes[1].set_title("Tümör alanı dağılımı (tümörlü kesitler)")
    axes[1].set_xlabel("Tümör alanı (piksel)")
    axes[1].set_ylabel("Kesit sayısı")

    splits = ["train", "val", "test"]
    names = {"train": "Eğitim", "val": "Doğrulama", "test": "Test"}
    t = [stats["bolme"][s]["tumorlu_kesit"] for s in splits]
    n = [stats["bolme"][s]["kesit"] - stats["bolme"][s]["tumorlu_kesit"] for s in splits]
    x = np.arange(3)
    axes[2].bar(x, t, color="#c0392b", label="Tümörlü")
    axes[2].bar(x, n, bottom=t, color="#7f8c8d", label="Tümörsüz")
    axes[2].set_xticks(x, [f"{names[s]}\n({stats['bolme'][s]['hasta']} hasta)" for s in splits])
    axes[2].set_title("Hasta bazlı veri bölme")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "veri_dagilimi.png", dpi=150)
    plt.close()
    print(f"\nŞekiller kaydedildi: {config.FIG_DIR}")


if __name__ == "__main__":
    main()
