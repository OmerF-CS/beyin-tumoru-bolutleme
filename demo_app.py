"""Canlı demo: tarayıcıda çalışan beyin tümörü bölütleme arayüzü (Gradio).

Kullanım:
    python demo_app.py            -> tarayıcıda http://127.0.0.1:7860 açılır
    python demo_app.py --share    -> geçici herkese açık bağlantı da üretir

Sunum için: demo_ornekler/ klasöründeki görüntüler (test hastalarından,
modelin eğitimde hiç görmediği kesitler) arayüzün altında hazır örnek olarak
listelenir; tıklamanız yeterli.

NOT: Bu uygulama eğitim amaçlıdır; klinik tanıda kullanılamaz.
"""
import argparse
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

import config
from src.data_utils import normalize, read_image, read_mask, to_rgb
from src.metrics import dice, iou
from src.models import load_checkpoint
from src.otsu_baseline import otsu_segment
from src.visualize import overlay, prob_heatmap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}


def load_models():
    print(f"Cihaz: {DEVICE}")
    for m in config.MODEL_NAMES:
        ck = config.CHECKPOINT_DIR / f"{m}_best.pt"
        if ck.exists():
            try:
                MODELS[m] = load_checkpoint(ck, DEVICE)
                print(f"Yüklendi: {config.MODEL_LABELS[m]}")
            except Exception as e:
                print(f"UYARI: {ck.name} yüklenemedi ({e}); bu model listede gösterilmeyecek.")
    if not MODELS:
        print("UYARI: Eğitilmiş model bulunamadı; yalnızca Otsu kullanılabilir. Önce train.py çalıştırın.")


def choices():
    return [config.MODEL_LABELS[m] for m in MODELS] + [config.MODEL_LABELS["otsu"]]


def label_to_key(label):
    return next(k for k, v in config.MODEL_LABELS.items() if v == label)


# --------------------------------------------------------------------------
def get_input(image, file):
    """Yüklenen dosya varsa (ör. .tif) onu, yoksa görüntü kutusunu kullanır."""
    if file is not None:
        path = getattr(file, "name", file)
        return read_image(path)
    if image is None:
        return None
    return to_rgb(image)


def get_mask(mask, shape):
    if mask is None:
        return None
    m = mask if mask.ndim == 2 else mask[..., 0]
    m = (m > (127 if m.max() > 1 else 0)).astype(np.uint8)
    if m.shape != shape:
        m = cv2.resize(m, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return m


def run(img, key, threshold):
    """Seçilen yöntemi çalıştırır; (olasılık haritası, ikili maske, süre ms) döndürür."""
    t0 = time.perf_counter()
    if key == "otsu":
        pred = otsu_segment(img)
        prob = pred.astype(np.float32)
    else:
        model, ck = MODELS[key]
        size = ck["img_size"]
        x = cv2.resize(img, (size, size)) if img.shape[:2] != (size, size) else img
        x = torch.from_numpy(normalize(x, config.MEAN, config.STD))[None].to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
        if prob.shape != img.shape[:2]:
            prob = cv2.resize(prob, img.shape[1::-1])
        pred = (prob > threshold).astype(np.uint8)
    if pred.sum() < config.MIN_TUMOR_PIXELS:
        pred = np.zeros_like(pred)
    return prob, pred, 1000 * (time.perf_counter() - t0)


def segment(image, file, mask, model_label, threshold):
    img = get_input(image, file)
    if img is None:
        return None, None, "Lütfen bir MR görüntüsü yükleyin veya aşağıdaki örneklerden birini seçin."
    gt = get_mask(mask, img.shape[:2])
    key = label_to_key(model_label)
    prob, pred, ms = run(img, key, threshold)

    head = (img[..., 1] > 10).sum()
    area = int(pred.sum())
    found = area > 0
    lines = [
        f"### {'🔴 Tümör tespit edildi' if found else '🟢 Tümör tespit edilmedi'}",
        f"- **Yöntem:** {model_label}",
        f"- **Bölütlenen alan:** {area} piksel"
        + (f" (kafa bölgesinin %{100 * area / max(head, 1):.1f}'i)" if found else ""),
        f"- **En yüksek tümör olasılığı:** {float(prob.max()):.2f}",
        f"- **Çıkarım süresi:** {ms:.1f} ms ({'GPU' if DEVICE.type == 'cuda' and key != 'otsu' else 'CPU'})",
    ]
    if gt is not None:
        lines += [
            "",
            "**Uzman maskesiyle karşılaştırma**",
            f"- Dice: **{dice(pred, gt):.3f}**   |   IoU: **{iou(pred, gt):.3f}**",
            f"- Uzman maskesindeki alan: {int(gt.sum())} piksel",
        ]
    lines += ["", "_Kırmızı: model tahmini · Yeşil kontur: uzman maskesi · Eğitim amaçlıdır, tanı aracı değildir._"]
    return overlay(img, pred=pred, gt=gt), prob_heatmap(img, prob), "\n".join(lines)


def compare_all(image, file, mask, threshold):
    img = get_input(image, file)
    if img is None:
        return []
    gt = get_mask(mask, img.shape[:2])
    gallery = [(overlay(img, gt=gt) if gt is not None else img,
                "Uzman maskesi" if gt is not None else "Giriş görüntüsü")]
    for label in choices():
        _, pred, ms = run(img, label_to_key(label), threshold)
        cap = label + (f" | Dice {dice(pred, gt):.2f}" if gt is not None else "") + f" | {ms:.0f} ms"
        gallery.append((overlay(img, pred=pred, gt=gt), cap))
    return gallery


def demo_examples():
    rows = []
    for p in sorted(Path(config.DEMO_DIR).glob("*_goruntu.png")):
        mp = p.with_name(p.name.replace("_goruntu.png", "_maske.png"))
        rows.append([str(p), str(mp) if mp.exists() else None])
    return rows


# --------------------------------------------------------------------------
def build_ui():
    labels = choices()
    default = labels[0]
    with gr.Blocks(title="Beyin Tümörü Bölütleme") as demo:
        gr.Markdown(
            "# 🧠 MR Görüntülerinde Beyin Tümörü Bölütleme\n"
            "Bir FLAIR MR kesiti yükleyin (veya aşağıdan örnek seçin), yöntemi seçip **Bölütle**'ye basın. "
            "Uzman maskesi de verilirse Dice/IoU hesaplanır."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(type="numpy", label="MR görüntüsü (PNG/JPG)", height=300)
                inp_file = gr.File(label="veya veri setinden .tif dosyası", file_types=[".tif", ".tiff", ".png", ".jpg"])
                inp_mask = gr.Image(type="numpy", image_mode="L", label="Uzman maskesi (isteğe bağlı)", height=200)
                model_dd = gr.Dropdown(labels, value=default, label="Yöntem")
                thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Karar eşiği")
                with gr.Row():
                    btn = gr.Button("Bölütle", variant="primary")
                    btn_all = gr.Button("Tüm yöntemleri karşılaştır")
            with gr.Column(scale=2):
                with gr.Row():
                    out_ov = gr.Image(label="Bölütleme sonucu", height=360)
                    out_heat = gr.Image(label="Tümör olasılık haritası", height=360)
                out_info = gr.Markdown()
                gallery = gr.Gallery(label="Yöntem karşılaştırması", columns=5, height=300)

        btn.click(segment, [inp_img, inp_file, inp_mask, model_dd, thr], [out_ov, out_heat, out_info])
        btn_all.click(compare_all, [inp_img, inp_file, inp_mask, thr], gallery)

        ex = demo_examples()
        if ex:
            gr.Examples(ex, inputs=[inp_img, inp_mask],
                        label="Test hastalarından örnekler (model bu hastaları eğitimde görmedi)")
    return demo


def self_check():
    """Arayüzü açmadan demo hattını sınar (kontrol_et.py tarafından kullanılır)."""
    ui = build_ui()
    ex = demo_examples()
    if ex:
        img = read_image(ex[0][0])
        mask = read_mask(ex[0][1]) * 255 if ex[0][1] else None
    else:  # örnek yoksa yapay bir görüntü
        img = np.zeros((256, 256, 3), np.uint8)
        cv2.circle(img, (128, 128), 90, (90, 90, 90), -1)
        cv2.circle(img, (110, 120), 20, (120, 200, 140), -1)
        mask = None
    for label in choices():
        ov, heat, info = segment(img, None, mask, label, 0.5)
        assert ov is not None and ov.shape == img.shape and heat.shape == img.shape, label
        print(f"  {label}: OK ({info.splitlines()[0].replace('#', '').strip()})")
    gallery = compare_all(img, None, mask, 0.5)
    assert len(gallery) == len(choices()) + 1
    print(f"DEMO KONTROLÜ BAŞARILI ({len(choices())} yöntem, {len(ex)} örnek)")
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Geçici herkese açık bağlantı oluştur")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--kontrol", action="store_true", help="Arayüzü açmadan demo hattını sına ve çık")
    args = ap.parse_args()
    load_models()
    if args.kontrol:
        self_check()
    else:
        build_ui().launch(server_name="127.0.0.1", server_port=args.port, share=args.share,
                          inbrowser=True, allowed_paths=[str(config.DEMO_DIR)])
