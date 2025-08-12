@echo off
REM Quick API Key Setup for Een Unity Mathematics
REM Run this once to set up your API keys permanently

echo 🔑 Een Unity Mathematics - API Key Setup
echo.

REM Check if .env.local exists
if exist ".env.local" (
    echo ✅ .env.local already exists
    echo Edit .env.local to update your API keys
) else (
    echo 📝 Creating .env.local from template...
    copy .env.example .env.local
    echo.
    echo ⚠️  IMPORTANT: Edit .env.local and add your real API keys:
    echo    OPENAI_API_KEY=your-real-openai-key-here
    echo    ANTHROPIC_API_KEY=your-real-anthropic-key-here
    echo.
    echo Opening .env.local for editing...
    notepad .env.local
)

echo.
echo 🚀 After adding your API keys to .env.local:
echo    - Run: python api/unified_server.py
echo    - Or: python scripts/LAUNCH_FULL_EXPERIENCE.py
echo    - Your keys will be loaded automatically!
echo.
echo 🎭 Demo mode works without API keys if you prefer
pause