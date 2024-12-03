@echo off
echo Setting up global AI testing environment...

:: Step 1: Check and Install Python
echo Checking for Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Installing Python globally...
    powershell -Command "& {Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe -OutFile python-installer.exe; Start-Process python-installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1' -NoNewWindow -Wait; Remove-Item python-installer.exe}"
) else (
    echo Python is already installed.
)

:: Step 2: Upgrade pip globally
echo Upgrading pip globally...
python -m pip install --upgrade pip

:: Step 3: Install required Python libraries globally
echo Installing required Python libraries globally...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers

:: Step 4: Verify installations globally
echo Verifying global installations...
python -c "import torch, transformers; print('PyTorch Version:', torch.__version__, 'Transformers Version:', transformers.__version__)"

echo Global AI testing setup complete!
pause
