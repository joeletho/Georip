@echo off
SET CURR_DIR=%~dp0
SET CURR_DIR=%CURR_DIR:~0,-1%

FOR %%I IN ("%CURR_DIR%\..") DO SET ROOT_DIR=%%~fI
SET VENV_DIR=%ROOT_DIR%\.env

echo Install path: %ROOT_DIR%

REM Check if Python 3.10 is installed
python3.10 --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python 3.10 not found. Please install Python 3.10 and try again.
    exit /b 1
)

IF NOT EXIST "%VENV_DIR%" (
    echo Creating virtual environment...
    python3.10 -m venv "%VENV_DIR%"
)

echo Activating virtual environment...
CALL "%VENV_DIR%\Scripts\activate.bat"

echo Upgrading pip...
pip install --upgrade pip

IF EXIST "%CURR_DIR%\requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r "%CURR_DIR%\requirements.txt"
) ELSE (
    echo requirements.txt not found. Please ensure it is in the setup directory.
    deactivate
    exit /b 1
)

echo Installation complete.
echo Run 'CALL %VENV_DIR%\Scripts\activate.bat' to activate the virtual environment and 'deactivate' to deactivate it.
