@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
pythonw src\modler_v2.py
if %ERRORLEVEL% NEQ 0 (
    echo 启动失败，错误代码: %ERRORLEVEL%
    python src\modler_v2.py
    pause
)
