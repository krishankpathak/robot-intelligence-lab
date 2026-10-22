# Setup script for Windows PowerShell
Write-Host "Setting up ROBOT INTELLIGENCE LAB..." -ForegroundColor Cyan

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run test suite to verify
Write-Host "Running automated tests..." -ForegroundColor Yellow
python -m pytest tests -v

Write-Host "Setup complete! Run '.\scripts\run.ps1' to launch the dashboard." -ForegroundColor Green
