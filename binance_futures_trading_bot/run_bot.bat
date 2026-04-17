@echo off
setlocal EnableExtensions
REM Run from this folder so imports and .env resolve the same as manual runs.
REM For 24/7 on Windows (Task Scheduler), use run_bot_supervised.bat instead.
cd /d "%~dp0"

echo [%date% %time%] Starting trading bot (Python 3.12) ...
py -3.12 -m trading_bot.main
set "EXITCODE=%ERRORLEVEL%"
echo [%date% %time%] Bot stopped. Exit code: %EXITCODE%
endlocal & exit /b %EXITCODE%
