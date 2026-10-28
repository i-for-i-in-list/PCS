@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run install.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B start.py %*
set "PCS_STATUS=%errorlevel%"
pause
exit /b %PCS_STATUS%
