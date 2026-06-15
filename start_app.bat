@echo off
echo Launch the ESCO Knowledge Graph application...
echo.

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    streamlit run esco_kg_streamlit.py --server.headless false --server.port 8501 --server.address localhost
) else (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
)
