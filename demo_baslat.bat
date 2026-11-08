@echo off
chcp 65001 > nul
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
echo Demo baslatiliyor... Tarayici otomatik acilmazsa: http://127.0.0.1:7860
echo Kapatmak icin bu pencerede Ctrl+C tuslarina basin.
python demo_app.py %*
pause
