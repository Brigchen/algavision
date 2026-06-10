@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
pythonw src\sorta.py
