@echo off
title Gradio Demo - LLM Hallucination Detector
cd /d "%~dp0"
echo Starting Gradio Demo UI using project virtual environment...
.\venv\Scripts\python.exe src\demo.py
echo.
pause
