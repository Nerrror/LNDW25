@echo off
setlocal enableextensions enabledelayedexpansion

REM Load .env if exists
if exist "%~dp0..\..\.env" (
  for /f "usebackq delims=" %%a in ("%~dp0..\..\.env") do set "%%a"
)

REM Default port if not set
if "%PORT%"=="" set PORT=8501

REM Ensure Python can import from src
set "PYTHONPATH=%~dp0..\src;%PYTHONPATH%"

python -m streamlit run "%~dp0..\src\app.py" --server.port %PORT% --server.headless true
