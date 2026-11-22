@echo off
rem Start the meetnote web service. Port, host, API key and access code come from .env
rem next to this file (see .env.example). Extra args are passed through, e.g. --port 19000.
title meetnote server
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONWARNINGS=ignore
"%~dp0.venv\Scripts\meetnote.exe" serve %*
pause
