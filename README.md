# TransPy
A local Ai that will translate your Python File to C++ to run faster. This is not done yet but be patient. :3
Please use python 3.10
Run the following script plz: 

----------------------------------------------------------------------------------------------
@echo off
echo Setting up the environment for Python to C++ AI Translator...

:: Step 1: Install Chocolatey (if not installed)
echo Checking for Chocolatey...
choco -v >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Chocolatey...
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; Invoke-WebRequest https://chocolatey.org/install.ps1 -UseBasicParsing | Invoke-Expression"
    refreshenv
)

:: Step 2: Install Python 3.10+ using Chocolatey
echo Installing Python 3.10 or higher...
choco install python --version 3.10 -y

:: Step 3: Install CUDA Toolkit (Optional, for GPU support)
echo Installing NVIDIA CUDA Toolkit...
choco install cuda -y

:: Step 4: Upgrade pip and install Python dependencies
echo Upgrading pip and installing dependencies...
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers

:: Step 5: Verify installation
echo Verifying installation...
python -c "import torch; import transformers; print('PyTorch Version:', torch.__version__, 'Transformers Version:', transformers.__version__)"

echo Environment setup complete!
pause

------------------------------------------------------------------------------------------
WARNING WILL NOT WORK WITH NORMAL CODE: ONLY MATHEMATICS

