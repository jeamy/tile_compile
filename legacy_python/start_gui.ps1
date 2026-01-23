#requires -version 5.1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$GuiDir = Join-Path $RepoRoot "gui"

if (-not (Test-Path -LiteralPath $GuiDir -PathType Container)) {
  Write-Error "gui directory not found: $GuiDir"
  exit 1
}

function Require-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    Write-Error "Missing prerequisite: $Name"
    exit 1
  }
}

Require-Command "python"

Write-Host "Info: starting Qt6 GUI (Python)."

Set-Location $RepoRoot

# No auto-install: if PySide6 is missing, error out with a clear message.
try {
  python -c "import PySide6" | Out-Null
} catch {
  Write-Error "Python dependency missing: PySide6`nPlease install it in your environment (e.g. venv) before running the GUI."
  exit 1
}

python -m gui.main
