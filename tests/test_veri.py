"""Veri okuma, hasta bazlı bölme ve veri artırma testleri."""
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.data_utils import (augment, find_pairs, imread_any, imwrite_any, load_or_create_split,
                            normalize, patient_split, read_image, read_mask)
from src.otsu_baseline import otsu_segment
from src.metrics import dice
from src.sentetik import make_synthetic_dataset


class _TmpDir:
    """Türkçe karakterli geçici klasör (Unicode yol desteğini de sınar)."""

    def __enter__(self):
        self.path = Path(tempfile.mkdtemp(prefix="btb_test_")) / "çalışma_ğüşiö"
        self.path.mkdir()
        return self.path

    def __exit__(self, *a):
        shutil.rmtree(self.path.parent, ignore_errors=True)


def test_unicode_yolda_okuma_yazma():
    with _TmpDir() as d:
        img = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
        assert imwrite_any(d / "görüntü.png", img)
        back = imread_any(d / "görüntü.png")
        assert back is not None and np.array_equal(back, img)
        assert imread_any(d / "olmayan.png") is None


def test_veri_bulma_ve_okuma():
    with _TmpDir() as d:
        make_synthetic_dataset(d, n_patients=4, n_slices=5)
        df = find_pairs(d)
        assert len(df) == 20 and df.patient.nunique() == 4
        assert df.has_tumor.any() and (~df.has_tumor).any()
        img, m = read_image(df.image_path[0]), read_mask(df.mask_path[0])
        assert img.shape == (256, 256, 3) and img.dtype == np.uint8
        assert m.shape == (256, 256) and set(np.unique(m)) <= {0, 1}


def test_hasta_bazli_bolmede_sizinti_yok():
    with _TmpDir() as d:
        make_synthetic_dataset(d, n_patients=20, n_slices=3)
        df = patient_split(find_pairs(d), 0.15, 0.15, seed=42)
        # Her hasta yalnızca tek bir kümede olmalı.
        assert (df.groupby("patient").split.nunique() == 1).all()
        counts = df.groupby("split").patient.nunique()
        assert counts["test"] == 3 and counts["val"] == 3 and counts["train"] == 14
        # Aynı tohum -> aynı bölme.
        df2 = patient_split(find_pairs(d), 0.15, 0.15, seed=42)
        assert (df.split.values == df2.split.values).all()


def test_110_hasta_icin_78_16_16():
    import pandas as pd
    df = pd.DataFrame({"patient": [f"P{i:03d}" for i in range(110)]})
    c = patient_split(df, 0.15, 0.15, seed=42).split.value_counts()
    assert (c["train"], c["val"], c["test"]) == (78, 16, 16)


def test_bolme_dosyasi_yeniden_kullanilir():
    with _TmpDir() as d:
        make_synthetic_dataset(d, n_patients=10, n_slices=2)
        f = d / "results" / "split.csv"
        a = load_or_create_split(d, f, 0.2, 0.2, seed=1)
        assert f.exists()
        b = load_or_create_split(d, f, 0.2, 0.2, seed=999)  # farklı tohum; dosya kullanılmalı
        assert (a.sort_values("image_path").split.values == b.sort_values("image_path").split.values).all()
        # Dosyada bilgisayara özgü yol tutulmuyor.
        assert "image_path" not in open(f, encoding="utf-8").readline()


def test_veri_artirma_maskeyi_bozmaz():
    rng = np.random.default_rng(0)
    img = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    mask[100:150, 80:140] = 1
    for _ in range(20):
        a, m = augment(img, mask, rng)
        assert a.shape == img.shape and a.dtype == np.uint8
        assert m.shape == mask.shape and set(np.unique(m)) <= {0, 1}
        assert a.flags["C_CONTIGUOUS"] and m.flags["C_CONTIGUOUS"]


def test_normalizasyon_bicimi():
    x = normalize(np.full((8, 8, 3), 128, np.uint8), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    assert x.shape == (3, 8, 8) and x.dtype == np.float32


def test_otsu_sentetik_tumoru_bulur():
    with _TmpDir() as d:
        make_synthetic_dataset(d, n_patients=1, n_slices=10)
        df = find_pairs(d)
        row = df[df.has_tumor].iloc[0]
        pred = otsu_segment(read_image(row.image_path))
        assert pred.shape == (256, 256)
        assert dice(pred, read_mask(row.mask_path)) > 0.5
