@echo off
echo Restarting Claude Code...
taskkill /f /im claude-code.exe >nul 2>&1
timeout /t 3 >nul
cd /d "C:\Users\Nouri\Documents\GitHub\Een"
echo Starting Claude Code in Een project...
claude-code