"""Model eğitimi.

Örnekler:
    python train.py --model unet
    python train.py --model attention_unet --epochs 50
    python train.py --model resnet34_unet --lr 3e-4 --batch-size 16

En iyi model (doğrulama Dice'ına göre) checkpoints/<model>_best.pt olarak,
eğitim geçmişi results/egitim_gecmisi_<model>.csv olarak kaydedilir.
"""
import argparse
import contextlib
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.data_utils import load_or_create_split
from src.dataset import BrainMRIDataset, worker_init_fn
from src.losses import DiceBCELoss
from src.models import build_model, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_context(device, use_amp):
    """GPU'da float16 karma hassasiyet; CPU'da boş bağlam (uyarı üretmez)."""
    if use_amp:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def train_one_epoch(model, loader, loss_fn, opt, scaler, device, use_amp):
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="  eğitim", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with amp_context(device, use_amp):
            logits = model(x)
            loss = loss_fn(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp, thr=0.5):
    """Doğrulama kaybı ve iki Dice ölçüsü döndürür:
    - global_dice: tüm doğrulama piksellerinde toplu Dice (model seçimi için)
    - tumor_slice_dice: yalnızca tümörlü kesitlerde ortalama kesit Dice'ı
    """
    model.eval()
    total, n = 0.0, 0
    tp = fp = fn = 0.0
    slice_dices = []
    for x, y in tqdm(loader, desc="  doğrulama", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_context(device, use_amp):
            logits = model(x)
            loss = loss_fn(logits.float(), y)
        total += loss.item() * x.size(0)
        n += x.size(0)
        pred = (torch.sigmoid(logits.float()) > thr).float()
        tp += (pred * y).sum().item()
        fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item()
        inter = (pred * y).sum((1, 2, 3))
        denom = pred.sum((1, 2, 3)) + y.sum((1, 2, 3))
        has = y.sum((1, 2, 3)) > 0
        if has.any():
            slice_dices.extend((2 * inter[has] / denom[has]).tolist())
    global_dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return total / n, global_dice, float(np.mean(slice_dices)) if slice_dices else 0.0


def plot_history(hist: pd.DataFrame, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.epoch, hist.train_loss, label="Eğitim")
    axes[0].plot(hist.epoch, hist.val_loss, label="Doğrulama")
    axes[0].set_xlabel("Epok")
    axes[0].set_ylabel("Kayıp (Dice + BCE)")
    axes[0].set_title("Kayıp eğrileri")
    axes[0].legend()
    axes[1].plot(hist.epoch, hist.val_global_dice, label="Global Dice")
    axes[1].plot(hist.epoch, hist.val_tumor_slice_dice, label="Tümörlü kesit Dice")
    axes[1].set_xlabel("Epok")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Doğrulama Dice")
    axes[1].legend()
    fig.suptitle(config.MODEL_LABELS.get(model_name, model_name))
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / f"egitim_{model_name}.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Beyin tümörü bölütleme modeli eğitimi")
    ap.add_argument("--model", choices=config.MODEL_NAMES, default="unet")
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--patience", type=int, default=10, help="Erken durdurma sabrı (epok)")
    ap.add_argument("--workers", type=int, default=2, help="Windows'ta sorun olursa 0 yapın")
    ap.add_argument("--no-amp", action="store_true", help="Karma hassasiyeti (AMP) kapat")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="ResNet34 için ImageNet ağırlıklarını indirme (çevrimdışı test için)")
    ap.add_argument("--max-train-patients", type=int, default=None,
                    help="Hızlı test: eğitimde en fazla bu kadar hasta kullan (doğrulama da orantılı küçülür)")
    args = ap.parse_args()

    set_seed(args.seed)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Cihaz: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cpu":
        print("UYARI: GPU bulunamadı; eğitim CPU'da yapılacak ve çok yavaş olacak. "
              "GPU varsa CUDA'lı PyTorch kurulumunu kontrol edin (README).")

    df = load_or_create_split(
        args.data_dir, config.SPLIT_FILE, config.VAL_RATIO, config.TEST_RATIO, config.SEED
    )
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if args.max_train_patients:
        rng = np.random.default_rng(args.seed)
        tp = sorted(tr.patient.unique())
        vp = sorted(va.patient.unique())
        tp = rng.choice(tp, min(args.max_train_patients, len(tp)), replace=False)
        vp = rng.choice(vp, max(1, min(len(vp), args.max_train_patients // 4)), replace=False)
        tr, va = tr[tr.patient.isin(tp)], va[va.patient.isin(vp)]
        print(f"HIZLI TEST MODU: {len(tp)} eğitim, {len(vp)} doğrulama hastası kullanılıyor.")
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("HATA: eğitim veya doğrulama kümesi boş. Veri klasörünü kontrol edin.")
    print(f"Eğitim: {len(tr)} kesit ({tr.patient.nunique()} hasta) | "
          f"Doğrulama: {len(va)} kesit ({va.patient.nunique()} hasta)")

    dl_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                 worker_init_fn=worker_init_fn if args.workers > 0 else None,
                 persistent_workers=args.workers > 0)
    train_dl = DataLoader(BrainMRIDataset(tr, args.img_size, config.MEAN, config.STD, train=True),
                          batch_size=args.batch_size, shuffle=True,
                          drop_last=len(tr) > args.batch_size, **dl_kw)
    val_dl = DataLoader(BrainMRIDataset(va, args.img_size, config.MEAN, config.STD, train=False),
                        batch_size=args.batch_size, shuffle=False, **dl_kw)

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    n_params = count_parameters(model)
    print(f"Model: {config.MODEL_LABELS[args.model]} | Parametre: {n_params / 1e6:.2f} M")

    loss_fn = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = make_scaler(use_amp)

    ckpt_path = config.CHECKPOINT_DIR / f"{args.model}_best.pt"
    hist_path = config.RESULTS_DIR / f"egitim_gecmisi_{args.model}.csv"
    best, wait, history = -1.0, 0, []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_dl, loss_fn, opt, scaler, device, use_amp)
        va_loss, va_gdice, va_sdice = validate(model, val_dl, loss_fn, device, use_amp)
        sched.step()
        lr = opt.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                            val_global_dice=va_gdice, val_tumor_slice_dice=va_sdice,
                            lr=lr, sure_sn=time.time() - t0))
        pd.DataFrame(history).to_csv(hist_path, index=False)

        mark = ""
        if va_gdice > best:
            best, wait, mark = va_gdice, 0, "  <- en iyi, kaydedildi"
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "img_size": int(args.img_size),
                "epoch": int(epoch),
                "val_global_dice": float(va_gdice),
                "val_tumor_slice_dice": float(va_sdice),
                "n_params": int(n_params),
            }, ckpt_path)
        else:
            wait += 1
        print(f"Epok {epoch:3d}/{args.epochs} | kayıp {tr_loss:.4f}/{va_loss:.4f} | "
              f"val Dice (global) {va_gdice:.4f} | val Dice (tümörlü kesit) {va_sdice:.4f} | "
              f"{time.time() - t0:.0f} sn{mark}")
        if wait >= args.patience:
            print(f"Erken durdurma: {args.patience} epoktur iyileşme yok.")
            break

    plot_history(pd.DataFrame(history), args.model)
    print(f"\nToplam süre: {(time.time() - t_start) / 60:.1f} dk | En iyi val Dice: {best:.4f}")
    print(f"Model: {ckpt_path}")


if __name__ == "__main__":
    main()
