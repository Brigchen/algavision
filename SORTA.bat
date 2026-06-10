@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
pythonw src\sorta.py
if %ERRORLEVEL% NEQ 0 (
    echo 启动失败，错误代码: %ERRORLEVEL%
    python src\sorta.py
    pause
)
