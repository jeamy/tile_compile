$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PayloadDir = Join-Path $ScriptDir "payload"
$InstallRoot = Join-Path $env:USERPROFILE "tilecompile"
$LogDir = Join-Path $InstallRoot "logs"
$RunsDir = Join-Path $InstallRoot "runs"
$Port = if ($env:TILE_COMPILE_GUI2_PORT) { [int]$env:TILE_COMPILE_GUI2_PORT } else { 8080 }
$HostName = "127.0.0.1"
$Url = "http://${HostName}:${Port}/ui/"
$BackendBin = Join-Path $InstallRoot "web_backend_cpp\build\tile_compile_web_backend.exe"

function Write-Info($Message) {
  Write-Host "[gui2] $Message"
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
  
  # Check if this is an update (installation already exists)
  $IsUpdate = $false
  if ((Test-Path (Join-Path $InstallRoot "web_backend_cpp")) -or 
      (Test-Path (Join-Path $InstallRoot "web_frontend")) -or 
      (Test-Path (Join-Path $InstallRoot "tile_compile_cpp"))) {
    $IsUpdate = $true
    Write-Info "Existierende Installation gefunden - fuehre selektives Update durch"
  } else {
    Write-Info "Neue Installation - kopiere alle Dateien"
  }
  
  if ($IsUpdate) {
    # Selective update: only replace app directories, preserve user data
    # Remove old app directories
    $AppDirs = @("web_frontend", "web_backend_cpp", "tile_compile_cpp")
    foreach ($dir in $AppDirs) {
      $targetDir = Join-Path $InstallRoot $dir
      if (Test-Path $targetDir) {
        Remove-Item -Path $targetDir -Recurse -Force
      }
    }
    
    # Copy only app directories from payload
    foreach ($dir in $AppDirs) {
      $sourceDir = Join-Path $PayloadDir $dir
      $targetDir = Join-Path $InstallRoot $dir
      if (Test-Path $sourceDir) {
        $null = robocopy $sourceDir $targetDir /E /NFL /NDL /NJH /NJS /NP
        if ($LASTEXITCODE -ge 8) {
          throw "robocopy fehlgeschlagen fuer $dir (ExitCode=$LASTEXITCODE)"
        }
      }
    }
    
    Write-Info "App-Dateien aktualisiert. User-Daten (configs, runs, astap, pcc) bleiben erhalten."
  } else {
    # Fresh install: mirror everything
    $null = robocopy $PayloadDir $InstallRoot /MIR /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) {
      throw "robocopy fehlgeschlagen (ExitCode=$LASTEXITCODE)"
    }
  }
}

if (-not (Test-Path $PayloadDir)) {
  throw "payload\ fehlt."
}

Sync-Payload
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
New-Item -ItemType Directory -Path $RunsDir -Force | Out-Null

if (-not (Test-Path $BackendBin)) {
  throw "Backend-Binary nicht gefunden: $BackendBin"
}

$env:TILE_COMPILE_PROJECT_ROOT = $InstallRoot
$env:TILE_COMPILE_HOST = $HostName
$env:TILE_COMPILE_PORT = "$Port"
$env:TILE_COMPILE_CLI = Join-Path $InstallRoot "tile_compile_cpp\build\tile_compile_cli.exe"
$env:TILE_COMPILE_RUNNER = Join-Path $InstallRoot "tile_compile_cpp\build\tile_compile_runner.exe"
$env:TILE_COMPILE_RUNS_DIR = $RunsDir
$env:TILE_COMPILE_CONFIG = Join-Path $InstallRoot "tile_compile_cpp\tile_compile.yaml"
$env:TILE_COMPILE_SCHEMA = Join-Path $InstallRoot "tile_compile_cpp\tile_compile.schema.yaml"
$env:TILE_COMPILE_PRESETS_DIR = Join-Path $InstallRoot "tile_compile_cpp\examples"
$env:TILE_COMPILE_UI_DIR = Join-Path $InstallRoot "web_frontend"
$AllowedRoots = @($InstallRoot, $env:USERPROFILE)
if ($env:TEMP) { $AllowedRoots += $env:TEMP }
if ($env:TMP -and $env:TMP -ne $env:TEMP) { $AllowedRoots += $env:TMP }
$env:TILE_COMPILE_ALLOWED_ROOTS = ($AllowedRoots | Where-Object { $_ } | Select-Object -Unique) -join ";"
# Optionale Backend-Memory-Guard Overrides:
# TILE_COMPILE_BACKEND_SUBPROCESS_CAPTURE_BYTES (default 1048576)
# TILE_COMPILE_BACKEND_JOB_STDIO_STORE_BYTES (default 131072)
# TILE_COMPILE_BACKEND_SCAN_FRAMES_PREVIEW (default 256)
# TILE_COMPILE_BACKEND_SCAN_PER_DIR_FRAMES_PREVIEW (default 32)
# TILE_COMPILE_BACKEND_SCAN_PER_DIR_RESULTS_PREVIEW (default 64)
# TILE_COMPILE_BACKEND_SCAN_MESSAGES_PREVIEW (default 128)
# TILE_COMPILE_BACKEND_SCAN_COLOR_CANDIDATES_PREVIEW (default 32)
# TILE_COMPILE_BACKEND_REPORT_EVENTS_MAX (default 4096)
# TILE_COMPILE_BACKEND_REPORT_LOG_TAIL (default 128)
# TILE_COMPILE_BACKEND_REPORT_TEXT_BYTES (default 262144)
# TILE_COMPILE_BACKEND_REPORT_JSON_FILE_BYTES (default 4194304)
# TILE_COMPILE_BACKEND_RETAINED_JOBS (default 128)
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

Write-Info "Starte Crow-Backend im Vordergrund auf $Url (Ctrl+C zum Beenden)."
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

$backendProcess = $null
$exitCode = 0
try {
  $backendProcess = Start-Process -FilePath $BackendBin -WorkingDirectory $InstallRoot -NoNewWindow -PassThru
  Write-Info "Crow-Backend laeuft mit PID $($backendProcess.Id)."
  Wait-Process -Id $backendProcess.Id
  $backendProcess.Refresh()
  $exitCode = $backendProcess.ExitCode
} finally {
  if ($backendProcess) {
    $backendProcess.Refresh()
    if (-not $backendProcess.HasExited) {
      Write-Info "Beende Crow-Backend."
      Stop-Process -Id $backendProcess.Id
      Wait-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue
    }
  }
}

if ($exitCode -ne 0) {
  throw "Backend-Prozess mit ExitCode $exitCode beendet."
}
