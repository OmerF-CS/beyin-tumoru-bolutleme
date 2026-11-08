@echo off
chcp 65001 > nul
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
REM Tum deney hattini sirayla calistirir: veri kesfi, 3 modelin egitimi, degerlendirme.

if exist checkpoints\*.pt (
    echo UYARI: checkpoints klasorunde egitilmis modeller var; tam egitim bunlarin UZERINE YAZAR.
    echo Iptal etmek icin bu pencereyi kapatin, devam etmek icin bir tusa basin.
    pause > nul
)

echo === 1/5 Veri kesfi ===
python explore_data.py || goto :error

echo === 2/5 U-Net (sifirdan) egitimi ===
python train.py --model unet --epochs 40 || goto :error

echo === 3/5 Attention U-Net egitimi ===
python train.py --model attention_unet --epochs 40 || goto :error

echo === 4/5 ResNet34 kodlayicili U-Net egitimi ===
python train.py --model resnet34_unet --epochs 40 || goto :error

echo === 5/5 Test kumesinde degerlendirme ===
python evaluate.py || goto :error

echo.
echo Bitti! Sonuclar: results\    Demo icin: demo_baslat.bat
pause
goto :eof

:error
echo HATA olustu, yukaridaki mesaji kontrol edin.
pause
exit /b 1
