@echo off
title LLM Hallucination Detection - Advanced Dashboard
cd /d "%~dp0"
echo ===================================================
echo Starting Interactive Hallucination Audit Dashboard
echo Open Browser: http://127.0.0.1:7860
echo ===================================================
.\venv\Scripts\python.exe src\demo_advanced.py
pause
