# PowerShell script to launch the Streamlit application
# Usage: .\start_app.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ESCO Knowledge Graph - Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if the virtual environment exists
if (Test-Path ".\.venv\Scripts\python.exe") {
    Write-Host "Virtual environment found." -ForegroundColor Green
    
    # Check if Streamlit is installed
    $streamlitCheck = & ".\\.venv\Scripts\python.exe" -m pip show streamlit 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Streamlit not found. Installing dependencies..." -ForegroundColor Yellow
        & ".\\.venv\Scripts\python.exe" -m pip install -r requirements.txt
    }
    
    Write-Host "Launch application..." -ForegroundColor Green
    Write-Host ""
    
    # Go to the project directory
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location $scriptPath
    
    # Launch Streamlit directly from Python within the virtual environment
    & ".\\.venv\Scripts\python.exe" -m streamlit run esco_kg_streamlit.py --server.headless false --server.port 8501 --server.address localhost
} else {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run the setup first:" -ForegroundColor Yellow
    Write-Host "  .\setup.bat" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or manually:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
}

