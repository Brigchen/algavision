@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
pythonw src\modler_v2.py
