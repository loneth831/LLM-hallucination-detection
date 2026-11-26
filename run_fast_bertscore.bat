@echo off
title Fast BERTScore - LLM Hallucination Detector
cd /d "%~dp0"
echo Starting Fast BERTScore Detector using project virtual environment...
.\venv\Scripts\python.exe src\fast_bertscore.py
echo.
pause
