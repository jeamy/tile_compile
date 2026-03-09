$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PayloadDir = Join-Path $ScriptDir "payload"
$InstallRoot = Join-Path $env:USERPROFILE "tilecompile"
$VenvDir = Join-Path $InstallRoot ".venv"
$LogDir = Join-Path $InstallRoot "logs"
$RunsDir = Join-Path $InstallRoot "runs"
$Port = if ($env:TILE_COMPILE_GUI2_PORT) { [int]$env:TILE_COMPILE_GUI2_PORT } else { 8080 }
$HostName = "127.0.0.1"
$Url = "http://${HostName}:${Port}/ui/"

function Write-Info($Message) {
  Write-Host "[gui2] $Message"
}

function Test-PythonCommand {
  param([string[]]$PythonCommand)
  try {
    if ($PythonCommand.Length -gt 1) {
      & $PythonCommand[0] $PythonCommand[1..($PythonCommand.Length - 1)] -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
    } else {
      & $PythonCommand[0] -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
    }
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Get-PythonCommand {
  $candidates = @(
    @("py", "-3.11"),
    @("py", "-3"),
    @("python")
  )
  foreach ($candidate in $candidates) {
    $exe = $candidate[0]
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
      if (Test-PythonCommand -PythonCommand $candidate) {
        return $candidate
      }
    }
  }
  return $null
}

function Install-PythonIfMissing {
  $choice = $Host.UI.PromptForChoice(
    "Python fehlt",
    "Python 3.11+ wurde nicht gefunden. Jetzt installieren? Ohne Python starten GUI2-Backend und Reports nicht.",
    [System.Management.Automation.Host.ChoiceDescription[]]@(
      (New-Object System.Management.Automation.Host.ChoiceDescription "&Ja"),
      (New-Object System.Management.Automation.Host.ChoiceDescription "&Nein")
    ),
    1
  )
  if ($choice -ne 0) {
    return $false
  }
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    return $true
  }
  Write-Info "Python 3.11+ nicht gefunden und winget ist nicht verfuegbar."
  return $false
}

function Test-ServerReady {
  try {
    $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
    return $response.StatusCode -lt 500
  } catch {
    return $false
  }
}

function Open-BrowserIfEnabled {
  if ($env:TILE_COMPILE_GUI2_NO_BROWSER -eq "1") {
    return
  }
  Start-Process $Url
}

function Sync-Payload {
  New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
  $null = robocopy $PayloadDir $InstallRoot /MIR /NFL /NDL /NJH /NJS /NP
  if ($LASTEXITCODE -ge 8) {
    throw "robocopy fehlgeschlagen (ExitCode=$LASTEXITCODE)"
  }
}

function Invoke-Python {
  param(
    [string[]]$PythonCommand,
    [string[]]$Arguments
  )
  if ($PythonCommand.Length -gt 1) {
    & $PythonCommand[0] $PythonCommand[1..($PythonCommand.Length - 1)] @Arguments
  } else {
    & $PythonCommand[0] @Arguments
  }
  if ($LASTEXITCODE -ne 0) {
    throw "Python-Kommando fehlgeschlagen: $($PythonCommand -join ' ') $($Arguments -join ' ')"
  }
}

if (-not (Test-Path $PayloadDir)) {
  throw "payload\ fehlt."
}

$PythonCommand = Get-PythonCommand
if (-not $PythonCommand) {
  $installed = Install-PythonIfMissing
  if (-not $installed) {
    throw "Start abgebrochen: Python wurde nicht installiert. Die App funktioniert ohne Python nicht."
  }
  $PythonCommand = Get-PythonCommand
}
if (-not $PythonCommand) {
  throw "Start abgebrochen: Python 3.11+ konnte nicht gefunden oder installiert werden. Die App funktioniert ohne Python nicht."
}

Sync-Payload
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
New-Item -ItemType Directory -Path $RunsDir -Force | Out-Null

if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
  Write-Info "Erzeuge virtuelle Umgebung unter $VenvDir"
  Invoke-Python -PythonCommand $PythonCommand -Arguments @("-m", "venv", $VenvDir)
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
Write-Info "Installiere Python-Requirements in $VenvDir"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade fehlgeschlagen." }
& $VenvPython -m pip install -r (Join-Path $InstallRoot "web_backend\requirements-backend.txt")
if ($LASTEXITCODE -ne 0) { throw "Requirements-Installation fehlgeschlagen." }

$env:TILE_COMPILE_CLI = Join-Path $InstallRoot "tile_compile_cpp\build\tile_compile_cli.exe"
$env:TILE_COMPILE_RUNNER = Join-Path $InstallRoot "tile_compile_cpp\build\tile_compile_runner.exe"
$env:TILE_COMPILE_RUNS_DIR = $RunsDir
$env:TILE_COMPILE_CONFIG_PATH = Join-Path $InstallRoot "tile_compile_cpp\tile_compile.yaml"
$env:TILE_COMPILE_STATS_SCRIPT = Join-Path $InstallRoot "tile_compile_cpp\scripts\generate_report.py"
$env:TILE_COMPILE_ALLOWED_ROOTS = "$InstallRoot;$env:USERPROFILE"
$env:PYTHONUNBUFFERED = "1"
$LibDir = Join-Path $InstallRoot "tile_compile_cpp\lib"
if (Test-Path $LibDir) {
  if ($env:PATH) {
    $env:PATH = "$LibDir;$env:PATH"
  } else {
    $env:PATH = $LibDir
  }
}

if (Test-ServerReady) {
  Write-Info "GUI2-Backend laeuft bereits."
  Open-BrowserIfEnabled
  exit 0
}

Write-Info "Starte FastAPI-Backend im Vordergrund auf $Url (Ctrl+C zum Beenden)."
$BrowserUrl = $Url
if ($env:TILE_COMPILE_GUI2_NO_BROWSER -ne "1") {
  Start-Job -ScriptBlock {
    param([string]$url)
    for ($i = 0; $i -lt 30; $i++) {
      try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -lt 500) {
          Start-Process $url
          return
        }
      } catch {}
      Start-Sleep -Seconds 1
    }
  } -ArgumentList $BrowserUrl | Out-Null
}

& $VenvPython -m uvicorn app.main:app --app-dir (Join-Path $InstallRoot "web_backend") --host $HostName --port "$Port"
if ($LASTEXITCODE -ne 0) {
  throw "Backend-Prozess mit ExitCode $LASTEXITCODE beendet."
}
