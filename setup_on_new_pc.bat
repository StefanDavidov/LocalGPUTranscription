@echo off
setlocal
title Transcription App - One-Time Setup

echo =======================================================
echo      TRANSCRIPTION APP - NEW PC SETUP
echo =======================================================
echo.

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or 3.11 from python.org
    echo and make sure to check "Add Python to PATH" during install.
    pause
    exit /b
)
echo [OK] Python found.

:: 2. Create Virtual Environment
if exist "venv" (
    echo [INFO] Virtual environment 'venv' already exists. Skipping creation.
) else (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b
    )
)

:: 3. Install Dependencies (PyTorch + CUDA)
echo [INFO] Installing dependencies (this may take a few minutes)...
echo         Downloading PyTorch with CUDA support...
call venv\Scripts\activate
:: Force upgrade pip first
python.exe -m pip install --upgrade pip

:: Install PyTorch specifically with CUDA 11.8 or 12.1 support ensuring compatibility
:: We use the extra-index-url to get the CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

:: 4. Verify Models
echo.
echo [INFO] Verifying models...
if exist "models\whisper" (
    echo [OK] Whisper models found in local folder.
) else (
    echo [INFO] Models folder missing. Running downloader...
    python preload_models.py
)

:: 5. Create Desktop Shortcut (Optional)
echo.
echo [INFO] Creating Desktop Shortcut...
set "SCRIPT_PATH=%~dp0launch.vbs"
set "ICON_PATH=%~dp0main.py"
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Transcription App.lnk"

powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT_PATH%');$s.TargetPath='%SCRIPT_PATH%';$s.WorkingDirectory='%~dp0';$s.IconLocation='python.exe,0';$s.Save()"

echo.
echo =======================================================
echo      SETUP COMPLETE!
echo =======================================================
echo.
echo You can now run the app using 'run.bat' or the Desktop shortcut.
echo.
pause
