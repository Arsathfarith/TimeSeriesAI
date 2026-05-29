$venvPath = Join-Path $PSScriptRoot 'venv'
if (-Not (Test-Path $venvPath)) {
    python -m venv $venvPath
}
$activate = Join-Path $venvPath 'Scripts\Activate.ps1'
Write-Host "To activate the environment, run:`n. $activate"
Write-Host 'Installing dependencies...'
& "$venvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$venvPath\Scripts\python.exe" -m pip install -r "$PSScriptRoot\requirements.txt"
Write-Host 'Setup complete. Run `python app.py` inside the venv.'
