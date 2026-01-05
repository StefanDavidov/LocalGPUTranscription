@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python preload_models.py
pause
