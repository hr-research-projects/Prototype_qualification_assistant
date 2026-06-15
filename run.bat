@echo off
REM Simple startup script for Windows CMD
REM Just run: run.bat

call .venv\Scripts\activate.bat
streamlit run esco_kg_streamlit.py --server.headless false --server.port 8501 --server.address localhost

