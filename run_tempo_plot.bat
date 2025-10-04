@echo off
echo Starting TEMPO Data Visualization...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Run the robust script
echo Running Python script...
python tempo_plot_robust.py

echo.
echo Script completed. Press any key to exit.
pause
