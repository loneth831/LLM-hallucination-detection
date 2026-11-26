@echo off
title BERTScore Simple - LLM Hallucination Detector
cd /d "%~dp0"
echo Starting BERTScore Simple Detector using project virtual environment...
.\venv\Scripts\python.exe src\bertscore_simple.py
echo.
pause
