# Test Rehberi (Değerlendirici için)

Bu rehber, projeyi kendi bilgisayarınızda farklı derinliklerde sınamanız için hazırlanmıştır. Her seviye bir öncekinden bağımsızdır; ihtiyacınız olan seviyeyi seçebilirsiniz.

| Seviye | Ne gösterir? | Veri seti | Süre |
|---|---|---|---|
| **0. Canlı demo** | Eğitilmiş modellerin test hastalarındaki çıktısı | Gerekmez | ~1 dk |
| **1. Otomatik kontrol** | Kurulum ve tüm kod hattının çalıştığı | Gerekmez | 2–5 dk |
| **2. Sonuçları yeniden üretme** | Rapordaki sayıların bu modellerden çıktığı | Gerekir | ~2 dk |
| **3. Hızlı eğitim testi** | Eğitim hattının gerçek veride çalıştığı | Gerekir | GPU'da ~5–10 dk |
| **4. Tam yeniden eğitim** | Tüm deneyin baştan tekrarı | Gerekir | GPU'da birkaç saate kadar |

**Gereksinimler:** Windows 10/11 (Linux/macOS için `.sh` betikleri ve README'deki komutlar), Python 3.11 veya 3.12, internet bağlantısı (yalnızca kurulum için). NVIDIA GPU, demo ve kontrol için gerekli değildir. GPU yoksa her şey CPU'da çalışır, yalnızca eğitim yavaşlar.

---

## İlk adım: Kurulum (bir kez)

Klasördeki **`kurulum.bat`** dosyasına çift tıklayın. Betik şunları yapar:

1. Proje klasöründe ayrı bir sanal ortam (`venv`) oluşturur. Sistemdeki Python paketlerine dokunmaz.
2. NVIDIA GPU varsa CUDA destekli PyTorch'u, yoksa CPU sürümünü kurar.
3. Diğer paketleri kurar ve sonunda otomatik kontrolü (Seviye 1, kısa sürüm) çalıştırır.

Kurulum internet hızına göre 5–15 dakika sürebilir. Projeyi kaldırmak için klasörü silmeniz yeterlidir.

## Seviye 0 – Canlı demo

**`demo_baslat.bat`** dosyasına çift tıklayın. Tarayıcıda `http://127.0.0.1:7860` açılır.

- Sayfanın altındaki **örnekler**, modelin eğitimde hiç görmediği test hastalarından alınmıştır. Birine tıklayıp **Bölütle**'ye basın.
- Kırmızı alan modelin tahminini, yeşil kontur uzman maskesini gösterir. Dice ve IoU değerleri sağda yazar.
- **Tüm yöntemleri karşılaştır** düğmesi, aynı kesitte dört yöntemi yan yana gösterir.
- Kendi görüntünüzü de yükleyebilirsiniz (PNG/JPG veya veri setindeki `.tif` dosyaları).

Demo yalnızca yerel bilgisayarda çalışır ve internete veri göndermez.

## Seviye 1 – Otomatik kontrol (veri seti gerekmez)

**`kontrol_et.bat`** dosyasına çift tıklayın (veya `python kontrol_et.py`). Betik, gerçek veriyle aynı dosya yapısında yapay bir MR veri seti üretir ve şunları sınar:

- **Paketler ve donanım:** sürümler ve GPU durumu.
- **Birim testleri (22 test):** Dice, IoU, HD95 ve kesinlik/duyarlılık metriklerinin bilinen değerleri; hasta bazlı bölmede sızıntı olmaması (hiçbir hasta iki kümede birden yer almaz) ve 110 hastanın 78/16/16 bölünmesi; veri artırmanın maskeyi bozmaması; Türkçe karakterli dosya yollarında okuma ve yazma; model çıkış boyutları ve **raporda verilen parametre sayıları** (U-Net 7.763.041, Attention U-Net 7.851.293); kayıp fonksiyonu ve geri yayılım; model kaydetme ve yükleme.
- **Uçtan uca hat:** veri keşfi, üç modelin kısa eğitimi, değerlendirme ve demo arayüzü (tarayıcı açılmadan).

Beklenen son satır: **`TÜM KONTROLLER BAŞARILI.`** Kontrol geçici bir klasörde yapılır; projedeki eğitilmiş modellere ve sonuçlara dokunmaz.

Birim testleri pytest ile de çalıştırılabilir: `python -m pytest`

## Seviye 2 – Rapordaki sonuçları yeniden üretme

1. Veri setini indirin: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation (ücretsiz Kaggle hesabı gerekir).
2. Zip'i proje içindeki `data/` klasörüne açın (iç içe klasör olması sorun değildir).
3. Komut satırında:
   ```
   venv\Scripts\activate
   python evaluate.py
   ```
4. `results/sonuclar.md` dosyasındaki tablo, rapordaki sonuç tablosuyla aynı olmalıdır.

Hangi hastaların test kümesinde olduğu `results/split.csv` dosyasında kayıtlıdır. `evaluate.py` bu dosyayı okuduğu için değerlendirme her bilgisayarda aynı 16 test hastasıyla yapılır.

## Seviye 3 – Hızlı eğitim testi

Veri seti `data/` klasöründeyken **`hizli_test.bat`** dosyasını çalıştırın. U-Net, 20 hasta üzerinde 3 epok eğitilir, test kümesinde değerlendirilir ve demo bu modelle açılır. Tüm çıktılar `hizli_test/` klasörüne yazılır; teslim edilen modeller değişmez. Kısa eğitildiği için bu modelin başarımı tam modelden düşüktür. Amaç, eğitim hattının gerçek veride çalıştığını göstermektir.

## Seviye 4 – Tam yeniden eğitim

`run_all.bat` üç modeli 40'ar epok eğitir ve değerlendirir. **Uyarı:** `checkpoints/` içindeki teslim edilmiş modellerin üzerine yazar (betik başlamadan önce onay ister). Teslim edilen modelleri korumak için klasörün bir kopyasında çalıştırın.

GPU'larda bazı işlemler deterministik olmadığından yeniden eğitilen modellerin sonuçları birkaç binde birlik farklılık gösterebilir. Rapordaki sayılar teslim edilen modellere aittir ve Seviye 2 ile birebir doğrulanabilir.

---

## Rapordaki iddialar nasıl doğrulanır?

| İddia | Doğrulama |
|---|---|
| Veri hasta bazında bölündü, sızıntı yok | `results/split.csv`; test: `test_hasta_bazli_bolmede_sizinti_yok` |
| Model parametre sayıları | test: `test_unet_cikis_boyutu_ve_parametre_sayisi` |
| Metrikler doğru hesaplanıyor | `tests/test_metrikler.py` |
| Sonuç tablosu | Seviye 2: `python evaluate.py` |
| Eğitim ayarları (öğrenme oranı, epok, kayıp) | `train.py --help`, `src/losses.py` |

## Sık karşılaşılan sorunlar

| Sorun | Çözüm |
|---|---|
| `kurulum.bat` "Python bulunamadı" diyor | Python 3.11/3.12 kurun, kurulumda "Add python.exe to PATH" seçeneğini işaretleyin |
| Demo sayfası açılmıyor | Tarayıcıda elle `http://127.0.0.1:7860` adresine gidin; port doluysa `demo_baslat.bat --port 7861` |
| Demoda yalnızca "Otsu" seçeneği var | `checkpoints/` klasöründe `.pt` dosyaları eksik; teslim paketinin tamamını açtığınızdan emin olun |
| GPU varken "GPU bulunamadı" uyarısı | PyTorch CPU sürümüyle kurulmuş; `venv` klasörünü silip `kurulum.bat`'ı tekrar çalıştırın |
| `CUDA out of memory` | `python train.py ... --batch-size 8` |
