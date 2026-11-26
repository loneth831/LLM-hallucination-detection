@echo off
title LLM Hallucination Detection - FastAPI Server
cd /d "%~dp0"
echo ===================================================
echo Starting FastAPI Hallucination Detection API
echo Swagger Documentation: http://127.0.0.1:8000/docs
echo ===================================================
.\venv\Scripts\python.exe src\api.py
pause
