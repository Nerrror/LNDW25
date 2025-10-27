@echo off
setlocal enableextensions enabledelayedexpansion

if exist .venv (
  echo [*] Virtual env already exists
) else (
  echo [*] Creating virtual env
  python -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo [*] Copying .env.example to .env if not present
if not exist .env copy .env.example .env >nul

echo [*] Done. Use:  call .venv\Scripts\activate  then  call scripts\run_streamlit.cmd
