@echo off
title Dashboard Admin RAG Pengaduan
echo ============================================
echo   Dashboard Admin - RAG Pengaduan
echo   Kabupaten Bandung
echo ============================================
echo.
echo Membuka dashboard di browser...
echo Tekan Ctrl+C untuk menghentikan server.
echo.

cd /d "%~dp0"
start "" http://localhost:8501
streamlit run admin_dashboard.py --server.port 8501 --server.headless true
pause
