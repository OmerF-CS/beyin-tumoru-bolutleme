"""Kurulum ve kod hattının otomatik kontrolü. VERİ SETİ GEREKMEZ.

Yapay (sentetik) bir MR veri seti üretir ve projenin tüm adımlarını onun
üzerinde uçtan uca çalıştırır:
    1. Paket ve GPU kontrolü
    2. Birim testleri (metrikler, veri bölme, veri artırma, Otsu, modeller)
    3. Uçtan uca hat: veri keşfi -> eğitim (kısa) -> değerlendirme
    4. Demo arayüzünün kontrolü (tarayıcı açılmadan)

Asıl modellerin ve sonuçların üzerine YAZMAZ; her şey geçici bir klasörde
yapılır ve sonunda silinir.

Kullanım:
    python kontrol_et.py            # tam kontrol (3 model), CPU'da birkaç dakika
    python kontrol_et.py --hizli    # yalnızca U-Net ile kısa kontrol
    python kontrol_et.py --sakla    # geçici klasörü silme (çıktıları incelemek için)
"""
import argparse
import importlib
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
if os.name == "nt":
    os.system("")  # Windows konsolunda renkleri etkinleştir
RESULTS = []


def report(name, status, detail="", sec=0.0):
    mark = {"ok": f"{GREEN}[ OK ]{RESET}", "fail": f"{RED}[HATA]{RESET}", "skip": f"{YELLOW}[ATLA]{RESET}"}[status]
    print(f"{mark} {name}" + (f"  ({sec:.1f} sn)" if sec >= 0.05 else "") + (f"\n       {detail}" if detail else ""))
    RESULTS.append((name, status))


def run_step(name, fn, *a):
    t0 = time.time()
    try:
        detail = fn(*a) or ""
        report(name, "ok", detail, time.time() - t0)
        return True
    except unittest.SkipTest as e:
        report(name, "skip", str(e))
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()
        report(name, "fail", f"{type(e).__name__}: {e}\n       " + "\n       ".join(tb[-6:]), time.time() - t0)
    return False


# ------------------------------------------------------------------ 1. paketler
def check_packages():
    print("\n=== 1. Paketler ve donanım ===")
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 9):
        report("Python sürümü", "fail", "Python 3.9 veya üstü gerekli (3.10+ önerilir)")
    required = [("numpy", "numpy"), ("pandas", "pandas"), ("cv2", "opencv-python"), ("scipy", "scipy"),
                ("skimage", "scikit-image"), ("matplotlib", "matplotlib"), ("tqdm", "tqdm"),
                ("torch", "torch"), ("gradio", "gradio"), ("segmentation_models_pytorch", "segmentation-models-pytorch")]
    missing = []
    for mod, pip_name in required:
        try:
            m = importlib.import_module(mod)
            report(f"{pip_name} {getattr(m, '__version__', '')}", "ok")
        except Exception as e:
            missing.append(pip_name)
            report(pip_name, "fail", f"yüklenemedi: {e}  ->  pip install {pip_name}")
    try:
        import torch
        if torch.cuda.is_available():
            report(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})", "ok")
        else:
            report("GPU", "skip", "CUDA'lı GPU bulunamadı; kod CPU'da çalışır (demo için yeterli, tam eğitim yavaş olur)")
    except Exception:
        pass
    return missing


# ------------------------------------------------------------------ 2. birim testleri
def run_unit_tests():
    print("\n=== 2. Birim testleri ===")
    for modname in ["tests.test_metrikler", "tests.test_veri", "tests.test_modeller"]:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            report(modname, "fail", f"modül yüklenemedi: {e}")
            continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("test_") and fn.__module__ == mod.__name__:
                run_step(f"{modname.split('.')[-1]}::{name}", fn)


# ------------------------------------------------------------------ 3. uçtan uca
def run_cmd(args, env):
    p = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"komut başarısız: python {' '.join(args)}\n{tail}")
    return p.stdout


def pipeline(tmp, models, have_gradio):
    print("\n=== 3. Uçtan uca hat (yapay veri) ===")
    from src.sentetik import make_synthetic_dataset
    data = tmp / "veri"
    work = tmp / "calisma"
    env = dict(os.environ, BTB_CALISMA_DIZINI=str(work), PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")

    ok = run_step("Yapay veri seti üretimi (14 hasta × 10 kesit)",
                  lambda: make_synthetic_dataset(data, n_patients=14, n_slices=10) and None)
    if not ok:
        return
    run_step("explore_data.py", lambda: run_cmd(["explore_data.py", "--data-dir", str(data)], env) and
             f"çıktı: {len(list((work / 'results' / 'figures').glob('*.png')))} şekil")
    for m in models:
        def train(m=m):
            out = run_cmd(["train.py", "--model", m, "--data-dir", str(data), "--epochs", "2", "--img-size", "128",
                           "--batch-size", "4", "--workers", "0", "--no-pretrained"], env)
            assert (work / "checkpoints" / f"{m}_best.pt").exists(), "checkpoint oluşmadı"
            last = [l for l in out.splitlines() if l.startswith("Epok")][-1]
            return last.split("|")[0].strip() + " tamamlandı, checkpoint kaydedildi"
        run_step(f"train.py --model {m} (2 epok, 128 px)", train)

    def evaluate():
        out = run_cmd(["evaluate.py", "--data-dir", str(data), "--batch-size", "8"], env)
        import pandas as pd
        s = pd.read_csv(work / "results" / "metrikler_ozet.csv")
        assert set(s.method) >= {"otsu", *models}, f"eksik yöntem: {set(s.method)}"
        figs = ["ornek_tahminler.png", "dice_dagilimi.png", "metrik_karsilastirma.png", "hata_ornekleri.png"]
        miss = [f for f in figs if not (work / "results" / "figures" / f).exists()]
        assert not miss, f"eksik şekil: {miss}"
        n_demo = len(list((work / "demo_ornekler").glob("*_goruntu.png")))
        assert n_demo > 0, "demo örnekleri oluşmadı"
        return f"{len(s)} yöntem değerlendirildi, {len(figs)} şekil, {n_demo} demo örneği"
    run_step("evaluate.py", evaluate)

    if have_gradio:
        run_step("demo_app.py --kontrol", lambda: run_cmd(["demo_app.py", "--kontrol"], env).strip().splitlines()[-1])
    else:
        report("demo_app.py --kontrol", "skip", "gradio kurulu değil")


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hizli", action="store_true", help="Yalnızca U-Net ile uçtan uca kontrol")
    ap.add_argument("--sakla", action="store_true", help="Geçici klasörü silme")
    args = ap.parse_args()
    t0 = time.time()
    print("Beyin Tümörü Bölütleme — kurulum ve kod kontrolü")

    missing = check_packages()
    run_unit_tests()

    if "torch" in missing:
        report("Uçtan uca hat", "skip", "PyTorch olmadan eğitim/değerlendirme sınanamaz")
    else:
        models = ["unet"] if args.hizli else ["unet", "attention_unet", "resnet34_unet"]
        if "segmentation-models-pytorch" in missing and "resnet34_unet" in models:
            models.remove("resnet34_unet")
        tmp = Path(tempfile.mkdtemp(prefix="btb_kontrol_"))
        try:
            pipeline(tmp, models, "gradio" not in missing)
        finally:
            if args.sakla:
                print(f"\nGeçici klasör korundu: {tmp}")
            else:
                shutil.rmtree(tmp, ignore_errors=True)

    n_ok = sum(s == "ok" for _, s in RESULTS)
    n_fail = sum(s == "fail" for _, s in RESULTS)
    n_skip = sum(s == "skip" for _, s in RESULTS)
    print("\n" + "=" * 60)
    print(f"Toplam: {n_ok} başarılı, {n_fail} hatalı, {n_skip} atlandı  ({time.time() - t0:.0f} sn)")
    if n_fail == 0:
        print(f"{GREEN}TÜM KONTROLLER BAŞARILI.{RESET} Proje bu bilgisayarda çalışmaya hazır.")
    else:
        print(f"{RED}Bazı kontroller başarısız.{RESET} Yukarıdaki [HATA] satırlarına bakın; "
              "çoğu zaman eksik paket kurulumu yeterlidir (README > Sorun giderme).")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
