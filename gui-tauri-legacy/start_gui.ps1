#requires -version 5.1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$GuiDir = $RepoRoot

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

Require-Command "npm"
Require-Command "cargo"
Require-Command "rustc"

Write-Host "Info: GUI workflow: Scan input first; if color mode is UNKNOWN you must confirm it before Start. The confirmation is stored as color_mode_confirmed in runs/<run_id>/run_metadata.json"

$NodeModules = Join-Path $GuiDir "node_modules"
if (-not (Test-Path -LiteralPath $NodeModules -PathType Container)) {
  Write-Error "node_modules missing in $GuiDir`nPlease run: cd gui-tauri-legacy; npm install"
  exit 1
}

Set-Location $GuiDir

# No auto-installations, just start dev server
npm run dev
