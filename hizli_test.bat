@echo off
chcp 65001 > nul
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat

REM Gercek veriyle hizli test: U-Net, 20 egitim hastasi, 3 epok.
REM Tum ciktilar hizli_test\ klasorune yazilir; asil modeller ve sonuclar DEGISMEZ.
set BTB_CALISMA_DIZINI=%~dp0hizli_test
if exist results\split.csv (
    if not exist hizli_test\results mkdir hizli_test\results
    copy /y results\split.csv hizli_test\results\split.csv > nul
)

echo === 1/3 Kisa egitim: U-Net ===
python train.py --model unet --epochs 3 --max-train-patients 20 --workers 0
if errorlevel 1 goto :error

echo === 2/3 Test kumesinde degerlendirme ===
python evaluate.py
if errorlevel 1 goto :error

echo === 3/3 Demo - hizli test modeli ile ===
echo Not: 3 epokluk model tam egitilmis modelden belirgin sekilde zayiftir; amac hattin calistigini gostermektir.
python demo_app.py
goto :eof

:error
echo HATA olustu, yukaridaki mesaji kontrol edin.
pause
exit /b 1
