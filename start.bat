@echo off
chcp 65001 >nul
cd /d "%~dp0"
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860 --open-browser
pause
