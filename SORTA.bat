@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
set "PYW=C:\ProgramData\Anaconda3\envs\py313\pythonw.exe"
set "PY=C:\ProgramData\Anaconda3\envs\py313\python.exe"
if not exist "%PYW%" set "PYW=pythonw"
if not exist "%PY%" set "PY=python"
"%PYW%" src\sorta.py
if errorlevel 1 (
    echo SORTA failed to start. ErrorLevel=%ERRORLEVEL%
    "%PY%" src\sorta.py
    pause
)
