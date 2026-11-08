"""Teslim paketi oluşturur: kod + eğitilmiş modeller + sonuçlar + demo örnekleri.

Paket sayesinde hoca/değerlendirici, veri setini indirmeden ve eğitim
yapmadan demoyu çalıştırabilir; veri setini indirirse evaluate.py ile
rapordaki sonuçları birebir yeniden üretebilir.

Kullanım (tüm eğitim ve evaluate.py bittikten sonra):
    python paketle.py
    python paketle.py --ekle Rapor.docx Sunum.pptx     # rapor ve sunumu da ekle

Çıktı: teslim/BeyinTumoruBolutleme_teslim.zip
Veri seti (data/), sanal ortam (venv/) ve hızlı test çıktıları pakete girmez.
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE_DIRS = {"data", "venv", ".venv", "hizli_test", "teslim", "__pycache__", ".pytest_cache", ".git", ".idea", ".vscode"}
CODE_SUFFIXES = {".py", ".md", ".txt", ".bat", ".sh", ".ini"}


def collect():
    files = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        top = rel.parts[0]
        if len(rel.parts) == 1 and p.suffix in CODE_SUFFIXES:
            files.append(rel)
        elif top in {"src", "tests"} and p.suffix in CODE_SUFFIXES:
            files.append(rel)
        elif top == "checkpoints" and p.suffix == ".pt":
            files.append(rel)
        elif top in {"results", "demo_ornekler"}:
            files.append(rel)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ekle", nargs="*", default=[], help="Pakete eklenecek ek dosyalar (rapor, sunum...)")
    ap.add_argument("--cikti", default=str(ROOT / "teslim" / "BeyinTumoruBolutleme_teslim.zip"))
    args = ap.parse_args()

    files = collect()
    ckpts = [f for f in files if f.parts[0] == "checkpoints"]
    demos = [f for f in files if f.parts[0] == "demo_ornekler"]
    print(f"Kod/doküman dosyası : {len([f for f in files if f.parts[0] not in {'checkpoints', 'results', 'demo_ornekler'}])}")
    print(f"Eğitilmiş model     : {len(ckpts)} -> {', '.join(f.name for f in ckpts) or 'YOK'}")
    print(f"Sonuç dosyası       : {len([f for f in files if f.parts[0] == 'results'])}")
    print(f"Demo örneği         : {len(demos)}")
    if not ckpts:
        print("UYARI: Eğitilmiş model yok. Önce train.py çalıştırın; aksi halde demo yalnızca Otsu ile çalışır.")
    if not demos:
        print("UYARI: Demo örneği yok. Önce evaluate.py çalıştırın.")

    out = Path(args.cikti)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, Path("BeyinTumoruBolutleme") / f)
        for extra in args.ekle:
            e = Path(extra)
            if e.exists():
                z.write(e, Path("BeyinTumoruBolutleme") / e.name)
                print(f"Eklendi: {e.name}")
            else:
                print(f"UYARI: bulunamadı, eklenmedi: {e}")
    print(f"\nPaket hazır: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("Hoca için: zip'i açıp önce kurulum.bat, ardından demo_baslat.bat (bkz. TEST_REHBERI.md)")


if __name__ == "__main__":
    main()
