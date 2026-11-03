@echo off
chcp 65001 > nul
echo ========================================================
echo  Сборка настольного приложения Windows (FastAPI + WebView)
echo ========================================================
echo.

if not exist venv (
    echo [1/3] Создание виртуального окружения...
    python -m venv venv
)

call venv\Scripts\activate
echo [2/3] Проверка и установка библиотек...
pip install -r requirements.txt

echo [3/3] Компиляция через PyInstaller...
pyinstaller --noconfirm --onedir --windowed ^
    --add-data "templates;templates" ^
    --name "DNS_Vibration_Monitor" ^
    desktop_app.py

echo.
echo ========================================================
echo  СБОРКА УСПЕШНО ЗАВЕРШЕНА!
echo  Исполняемый файл находится в папке:
echo  dist\DNS_Vibration_Monitor\DNS_Vibration_Monitor.exe
echo ========================================================
pause
