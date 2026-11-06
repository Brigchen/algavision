@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
set "PYW=C:\ProgramData\Anaconda3\envs\py313\pythonw.exe"
set "PY=C:\ProgramData\Anaconda3\envs\py313\python.exe"
if not exist "%PYW%" set "PYW=pythonw"
if not exist "%PY%" set "PY=python"
"%PYW%" src\modler_v2.py
if errorlevel 1 (
    echo MODLER failed to start. ErrorLevel=%ERRORLEVEL%
    "%PY%" src\modler_v2.py
    pause
)
