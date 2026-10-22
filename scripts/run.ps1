# Run script for Windows PowerShell
Write-Host "Launching ROBOT INTELLIGENCE LAB Streamlit Dashboard..." -ForegroundColor Cyan

if (Test-Path "venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
}

python -m streamlit run app.py
