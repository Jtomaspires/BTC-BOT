@echo off
setlocal EnableExtensions
REM Long-running wrapper: restarts Python if the process exits (crash, OOM, kill).
REM Use this as the Task Scheduler "Program" for 24/7 Windows VPS deployments.
REM Single-instance: do not run two copies; Task Scheduler should not overlap triggers.

cd /d "%~dp0"

if not exist "logs" mkdir logs

REM Seconds to wait before restarting after exit (network cooldown, API backoff).
if not defined BOT_RESTART_DELAY_SEC set "BOT_RESTART_DELAY_SEC=20"

:loop
call :log_line "===== Starting py -3.12 -m trading_bot.main ====="
py -3.12 -m trading_bot.main
set "EXITCODE=%ERRORLEVEL%"
call :log_line "Bot process exited with code %EXITCODE%. Restarting in %BOT_RESTART_DELAY_SEC%s ..."
timeout /t %BOT_RESTART_DELAY_SEC% /nobreak >nul
goto loop

:log_line
set "LINE=%~1"
>> "logs\supervisor.log" echo [%date% %time%] %LINE%
echo [%date% %time%] %LINE%
exit /b 0
