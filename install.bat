@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 -B install.py
) else (
    python -B install.py
)
set "PCS_STATUS=%errorlevel%"
pause
exit /b %PCS_STATUS%
