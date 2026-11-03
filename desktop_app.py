"""
Точка входа для запуска системы онлайн-мониторинга в виде
нативного настольного приложения Windows (без рамок браузера).
"""
import sys
import os
import threading
import time
import webview
import uvicorn
from app import app

def run_fastapi_server():
    """Запуск сервера FastAPI в фоновом потоке."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def main():
    # Запускаем бэкенд в отдельном потоке-демоне
    server_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    server_thread.start()

    # Даем серверу полсекунды на старт
    time.sleep(0.8)

    # Создаем нативное окно Windows
    window = webview.create_window(
        title="Система онлайн-мониторинга электродвигателей 0.4-6 кВ | ДНС Нефтедобыча",
        url="http://127.0.0.1:8000",
        width=1440,
        height=920,
        min_size=(1024, 700),
        resizable=True,
        confirm_close=True
    )

    # Запуск графического цикла (использует системный WebView2)
    webview.start()
    
    # При закрытии окна программа завершает работу
    sys.exit(0)

if __name__ == "__main__":
    main()
