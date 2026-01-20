@echo off
REM SAM2.1 Fine-tuning Environment Setup for Windows + RTX 3080
REM Run this script from Anaconda Prompt

echo ============================================
echo SAM2.1 Fine-tuning Environment Setup
echo ============================================
echo.

REM Create conda environment with Python 3.10
echo [1/4] Creating conda environment 'sam2_finetune' with Python 3.10...
call conda create -n sam2_finetune python=3.10 -y

REM Activate the environment
echo.
echo [2/4] Activating environment...
call conda activate sam2_finetune

REM Install PyTorch with CUDA 11.8
echo.
echo [3/4] Installing PyTorch 2.1.0 with CUDA 11.8...
call pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

REM Install additional dependencies
echo.
echo [4/4] Installing additional packages...
call pip install opencv-python matplotlib tensorboard tqdm numpy pillow

echo.
echo ============================================
echo Environment setup complete!
echo ============================================
echo.
echo To activate the environment, run:
echo     conda activate sam2_finetune
echo.
echo To test the environment, run:
echo     python scripts/step1_test_env.py
echo.

pause
