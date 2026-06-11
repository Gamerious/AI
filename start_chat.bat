@echo off
REM NEXUS-LM ChatGPT-style web app launcher (local, RTX 4060)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" serve_chat.py %*
pause
