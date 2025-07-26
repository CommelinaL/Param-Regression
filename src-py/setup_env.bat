@echo off
:: Check if conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda is not installed. Exiting...
    exit /b 1
)

:: Create the conda environment from environment.yml file
echo Creating conda environment from environment.yml...
conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo Error creating the conda environment. Exiting...
    exit /b 1
)

:: Extract environment name from the environment.yml file (first line after name:)
for /f "tokens=2 delims=:" %%a in ('findstr /i "name:" environment.yml') do set ENV_NAME=%%a

:: Remove spaces from ENV_NAME variable
set ENV_NAME=%ENV_NAME:~1%

:: Activate the conda environment
echo Activating conda environment %ENV_NAME%...
conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo Error activating the conda environment. Exiting...
    exit /b 1
)

:: Install additional dependencies from requirements.txt using pip
echo Installing pip dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing pip dependencies. Exiting...
    exit /b 1
)

echo Environment setup complete!