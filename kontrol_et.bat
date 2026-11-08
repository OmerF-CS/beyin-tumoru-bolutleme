@echo off
chcp 65001 > nul
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
REM Veri seti gerektirmeyen otomatik kontrol. Secenekler: --hizli  --sakla
python kontrol_et.py %*
pause
