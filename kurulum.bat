@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ==================================================
echo   Beyin Tumoru Bolutleme - Kurulum (Windows)
echo ==================================================

where python > nul 2>&1
if errorlevel 1 goto :nopython
python --version

if not exist venv\Scripts\activate.bat (
    echo Sanal ortam olusturuluyor: venv
    python -m venv venv
    if errorlevel 1 goto :error
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

REM ---- PyTorch: NVIDIA GPU varsa CUDA'li surum, yoksa CPU surumu ----
REM RTX 50xx serisi ekran kartlari icin asagidaki degeri cu128 yapin.
set CUDA_SURUM=cu126
nvidia-smi > nul 2>&1
if errorlevel 1 goto :cpu_torch

echo NVIDIA GPU bulundu. CUDA destekli PyTorch kuruluyor: %CUDA_SURUM%
pip install torch torchvision --index-url https://download.pytorch.org/whl/%CUDA_SURUM%
if errorlevel 1 goto :error
goto :others

:cpu_torch
echo NVIDIA GPU bulunamadi. PyTorch CPU surumu kuruluyor - demo icin yeterlidir.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :error

:others
pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Kurulum tamamlandi. Otomatik kontrol calistiriliyor...
echo.
python kontrol_et.py --hizli
echo.
echo Demo icin: demo_baslat.bat   ^|   Tam kontrol icin: kontrol_et.bat
pause
goto :eof

:nopython
echo HATA: Python bulunamadi.
echo https://www.python.org adresinden Python 3.11 veya 3.12 kurun ve
echo kurulumda "Add python.exe to PATH" secenegini isaretleyin.
pause
exit /b 1

:error
echo.
echo HATA: Kurulum basarisiz oldu. Yukaridaki mesaji kontrol edin (README - Sorun giderme).
pause
exit /b 1
