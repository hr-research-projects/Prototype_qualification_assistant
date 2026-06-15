@echo off
echo ========================================
echo ESCO Knowledge Graph - Setup
echo ========================================
echo.

echo Step 1: Create a virtual environment...
if exist .venv (
    echo The virtual environment already exists.
) else (
    python -m venv .venv
    echo Virtual environment created.
)

echo.
echo Step 2: Activate the virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Step 3: Upgrade pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Install the dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the application, run:
echo   streamlit run esco_kg_streamlit.py
echo.
echo Or use the startup script:
echo   start_app.bat
echo.
pause

