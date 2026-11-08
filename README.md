# MR Görüntülerinde Beyin Tümörü Bölütleme

Bilgisayarlı Görü dersi projesi. FLAIR MR kesitlerinde alt dereceli glioma (LGG) bölgesini piksel düzeyinde bölütleyen dört yöntem karşılaştırılır:

| Yöntem | Tür |
|---|---|
| Otsu eşikleme + morfoloji | Klasik görüntü işleme (öğrenmesiz, alt sınır) |
| U-Net (sıfırdan) | Derin öğrenme, rastgele başlatılmış |
| Attention U-Net | Dikkat kapılı U-Net |
| U-Net + ResNet34 | ImageNet ile önceden eğitilmiş kodlayıcı (transfer öğrenme) |

Proje bir de **canlı demo** içerir: tarayıcıda açılan arayüzde MR görüntüsü seçilir, tümör maskesi anında çizilir.

> **Değerlendirici iseniz:** Projeyi kendi bilgisayarınızda sınamak için **[TEST_REHBERI.md](TEST_REHBERI.md)** dosyasına bakın. Kısaca: `kurulum.bat`, ardından `demo_baslat.bat` veya `kontrol_et.bat`.

---

## Hızlı başlangıç (Windows)

| Ne yapmak istiyorsunuz? | Çift tıklayın | Veri seti gerekir mi? |
|---|---|---|
| Kurulum (bir kez) | `kurulum.bat` | Hayır |
| Her şeyin çalıştığını kontrol etme | `kontrol_et.bat` | Hayır |
| Canlı demo | `demo_baslat.bat` | Hayır (eğitilmiş modeller varsa) |
| Gerçek veriyle hızlı deneme (~10 dk) | `hizli_test.bat` | Evet |
| Tüm deney (3 modelin tam eğitimi) | `run_all.bat` | Evet |

Linux/macOS: `bash kurulum.sh`, ardından `source venv/bin/activate` ve aşağıdaki `python ...` komutları.

## 1. Klasör yapısı

```
beyin_tumoru_bolutleme/
├── config.py            # yollar ve varsayılan ayarlar
├── explore_data.py      # veri istatistikleri + örnek görseller
├── train.py             # model eğitimi
├── evaluate.py          # test kümesinde karşılaştırma, şekiller, demo örnekleri
├── demo_app.py          # canlı demo (Gradio)
├── kontrol_et.py        # veri gerektirmeyen otomatik kontrol
├── paketle.py           # teslim paketi (kod + modeller + sonuçlar) oluşturur
├── *.bat / *.sh         # kurulum ve başlatma betikleri
├── TEST_REHBERI.md      # değerlendirici için test rehberi
├── requirements.txt
├── src/
│   ├── data_utils.py    # veri okuma, hasta bazlı bölme, veri artırma
│   ├── dataset.py       # PyTorch Dataset
│   ├── models.py        # U-Net, Attention U-Net, ResNet34-U-Net
│   ├── losses.py        # Dice + BCE kaybı
│   ├── metrics.py       # Dice, IoU, HD95, kesinlik, duyarlılık
│   ├── otsu_baseline.py # klasik yöntem
│   ├── sentetik.py      # test için yapay veri üretici
│   └── visualize.py     # maske bindirme
├── tests/               # birim testleri (pytest)
├── data/                # veri seti buraya (siz ekleyeceksiniz)
├── checkpoints/         # eğitilen modeller (otomatik oluşur)
├── results/             # metrikler, şekiller, split.csv (otomatik oluşur)
└── demo_ornekler/       # demo için örnek görüntüler (otomatik oluşur)
```

## 2. Kurulum

**Kolay yol (Windows):** `kurulum.bat`. Sanal ortamı oluşturur, GPU varsa CUDA'lı PyTorch'u, yoksa CPU sürümünü kurar ve kısa bir kontrol çalıştırır.

**Elle kurulum:** Python 3.11 veya 3.12 önerilir.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# GPU destekli PyTorch: https://pytorch.org/get-started/locally/ adresinden
# sisteminize uygun komutu alın. Örnek (CUDA 12.6):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
python kontrol_et.py --hizli   # kurulumu doğrula
```

## 3. Veri seti

1. Kaggle'dan **LGG MRI Segmentation** veri setini indirin:
   https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
2. Zip dosyasını `data/` klasörüne açın. Sonuçta şuna benzer bir yapı olmalı:
   `data/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif`
   (İç içe klasör olması sorun değil; kod alt klasörleri kendisi tarar ve tekrar eden dosyaları ayıklar.)

Veri seti 110 hastadan 3929 FLAIR kesiti ve uzman tarafından çizilmiş maskeler içerir (Buda vd., 2019). Görüntüler 256×256 piksel ve 3 kanallıdır (kontrast öncesi, FLAIR, kontrast sonrası).

Başka bir klasördeyse tüm betiklere `--data-dir YOL` verebilirsiniz. Türkçe karakter içeren yollar (ör. `C:\Users\Ömer\...`) desteklenir.

## 4. Çalıştırma

```bash
python explore_data.py                    # veri istatistikleri ve örnek görseller
python train.py --model unet              # U-Net (sıfırdan)
python train.py --model attention_unet    # Attention U-Net
python train.py --model resnet34_unet     # ResNet34 kodlayıcılı U-Net
python evaluate.py                        # test kümesinde karşılaştırma
python demo_app.py                        # canlı demo
```

Önemli `train.py` seçenekleri:

| Seçenek | Varsayılan | Açıklama |
|---|---|---|
| `--epochs` | 40 | En fazla epok |
| `--batch-size` | 16 | Bellek yetmezse 8 veya 4 |
| `--lr` | 3e-4 | Başlangıç öğrenme oranı (kosinüs azaltma) |
| `--patience` | 10 | Erken durdurma sabrı |
| `--workers` | 2 | Windows'ta sorun olursa 0 |
| `--max-train-patients` | – | Hızlı test için eğitim hastası sınırı |
| `--no-pretrained` | – | ResNet34 için ImageNet ağırlığı indirme |

**Hasta bazlı bölme:** Hastalar %70 eğitim, %15 doğrulama, %15 test olarak ayrılır (78/16/16). Aynı hastanın kesitleri asla iki farklı kümeye düşmez; böylece komşu kesitlerin benzerliğinden doğan veri sızıntısı önlenir. Bölme `results/split.csv` dosyasına yalnızca hasta kimlikleriyle kaydedilir. Tüm modeller ve başka bilgisayarlar aynı bölmeyi kullanır.

## 5. Çıktılar

| Dosya | İçerik |
|---|---|
| `results/split.csv` | hasta → küme eşlemesi (tekrarlanabilirlik için) |
| `results/veri_istatistikleri.json` | hasta/kesit sayıları, tümör alanı istatistikleri |
| `results/figures/veri_ornekleri.png` | 3 MR kanalı ve uzman maskesi örnekleri |
| `results/figures/veri_dagilimi.png` | sınıf dağılımı, tümör alanı histogramı, veri bölme |
| `results/figures/egitim_<model>.png` | kayıp ve Dice eğrileri |
| `results/metrikler_ozet.csv` / `sonuclar.md` | **ana sonuç tablosu** |
| `results/metrikler_kesit.csv` | her test kesiti için metrikler |
| `results/figures/ornek_tahminler.png` | aynı kesitlerde tüm yöntemlerin görsel karşılaştırması |
| `results/figures/dice_dagilimi.png` | kesit ve hasta bazlı Dice kutu grafikleri |
| `results/figures/metrik_karsilastirma.png` | metriklerin çubuk grafiği |
| `results/figures/hata_ornekleri.png` | en iyi modelin zorlandığı kesitler |

### Metrikler

- **Dice (tümörlü kesit):** Uzman maskesinde tümör bulunan kesitlerde ortalama Dice. Ana metriktir.
- **Dice (hasta bazlı):** Hastanın tüm kesitleri birleştirilerek 3B hacim üzerinden hesaplanan Dice. Buda vd. (2019) ile karşılaştırmaya uygundur (orada ortalama %82 bildirilmiştir).
- **IoU:** Kesişim / birleşim.
- **HD95:** Sınırlar arası mesafelerin 95. yüzdeliği (piksel). Küçük olması daha iyidir.
- **Tespit duyarlılığı / özgüllüğü:** Kesit düzeyinde "tümör var mı?" sorusunun doğruluğu. Özgüllük, tümörsüz kesitlerde yanlış alarm üretmeme oranıdır.

## 6. Test ve doğrulama

```bash
python kontrol_et.py          # veri gerektirmeyen tam kontrol (yapay veriyle uçtan uca)
python kontrol_et.py --hizli  # yalnızca U-Net ile kısa kontrol
python -m pytest              # yalnızca birim testleri
```

`kontrol_et.py` yapay bir veri seti üretip paketleri, 22 birim testini, üç modelin kısa eğitimini, değerlendirmeyi ve demo arayüzünü sınar. Her şeyi geçici bir klasörde yapar; asıl modellere ve sonuçlara dokunmaz. Ayrıntılar için [TEST_REHBERI.md](TEST_REHBERI.md) dosyasına bakın.

## 7. Canlı demo (sunum için)

```bash
python demo_app.py        # veya demo_baslat.bat
```

Tarayıcıda `http://127.0.0.1:7860` açılır. Arayüzde neler var:

- Alttaki **örnekler** test hastalarındandır; model bu hastaları eğitimde hiç görmedi. Bir örneğe tıklayınca görüntü ve uzman maskesi yüklenir.
- **Bölütle:** seçili yöntemin sonucu (kırmızı), uzman maskesi (yeşil kontur), olasılık ısı haritası, Dice/IoU ve çıkarım süresi.
- **Tüm yöntemleri karşılaştır:** aynı görüntüde dört yöntem yan yana.
- **Karar eşiği** kaydırıcısı: eşiğin duyarlılık ile yanlış alarm arasındaki dengeyi nasıl değiştirdiğini canlı olarak göstermek için.

**Sunum günü kontrol listesi**

- [ ] Demoyu sunumdan önce aynı bilgisayarda bir kez çalıştırıp deneyin.
- [ ] Demo internet gerektirmez (`--share` kullanmayın).
- [ ] Olası bir aksaklığa karşı demonun 1 dakikalık ekran kaydını yedekte tutun.
- [ ] Gösterim için önerilen sıra: büyük tümörlü örnek → küçük tümörlü örnek → tümörsüz örnek → "Tüm yöntemleri karşılaştır" → eşik kaydırıcısı.

## 8. Teslim paketi

Tüm eğitim ve `evaluate.py` bittikten sonra:

```bash
python paketle.py --ekle Beyin_Tumoru_Bolutleme_Rapor.docx Beyin_Tumoru_Bolutleme_Sunum.pptx
```

`teslim/BeyinTumoruBolutleme_teslim.zip` oluşur. İçinde kod, eğitilmiş modeller, sonuçlar, `split.csv`, demo örnekleri ve (eklenirse) rapor ile sunum bulunur. Veri seti ve `venv` pakete girmez. Bu paketle değerlendirici veri indirmeden ve eğitim yapmadan demoyu çalıştırabilir.

## 9. Sorun giderme

| Sorun | Çözüm |
|---|---|
| `CUDA out of memory` | `--batch-size 8` veya `4` |
| Windows'ta DataLoader takılıyor veya hata veriyor | `--workers 0` |
| `*_mask.tif bulunamadı` | Veri setinin `data/` altına açıldığını kontrol edin veya `--data-dir` verin |
| GPU varken "GPU bulunamadı" | PyTorch CPU sürümüyle kurulmuş; `venv` klasörünü silip `kurulum.bat`'ı tekrar çalıştırın |
| ResNet34 ağırlıkları inmiyor | İnternet bağlantısını kontrol edin (yalnızca ilk eğitimde gerekir) |
| Demo portu dolu | `python demo_app.py --port 7861` |
| Demoda yalnızca "Otsu" var | `checkpoints/` boş; önce modelleri eğitin veya teslim paketini eksiksiz açın |

## Kaynaklar

- Buda, M., Saha, A., Mazurowski, M. A. (2019). Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. *Computers in Biology and Medicine*, 109, 218–225.
- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
- Oktay, O. vd. (2018). Attention U-Net: Learning where to look for the pancreas. *MIDL*.

> Bu proje eğitim amaçlıdır; klinik tanı için kullanılamaz.
