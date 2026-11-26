@echo off
title LLM Hallucination Detection - Evaluation Benchmark
cd /d "%~dp0"
echo ===================================================
echo Running Evaluation Benchmark
echo ===================================================
.\venv\Scripts\python.exe src\evaluate_benchmark.py
pause
