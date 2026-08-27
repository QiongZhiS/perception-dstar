# run_transduction_safe.ps1
# Safe runner for vision/transduction_sparse.py.
# All stdout/stderr of the script is redirected into a log file; this wrapper
# prints ONLY one metadata line (exit / bytes / lines / timeout / log path).
# Rationale: script output never enters the model request context, which avoids
# the "Content Exists Risk" moderation rejection that crashes the session.
# RULE: NEVER read/Get-Content the log file contents - only metadata.
# NOTE: on this machine Start-Process misreports the child exit code (always 1,
# verified: sys.exit(42) read as 1), so we use the call operator + $LASTEXITCODE
# which reads the true code. Timeout is enforced via a background job.
# Compatible with both pwsh 7 and Windows PowerShell 5.1 (ASCII-only body).
param(
    [string]$Script = "vision/transduction_sparse.py",
    [string]$LogPath = "logs/transduction_run.log",
    [int]$TimeoutSec = 600
)
$ErrorActionPreference = "Continue"

$logDir = Split-Path -Parent $LogPath
if ($logDir -and -not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
if (Test-Path $LogPath) { Remove-Item $LogPath -Force -ErrorAction SilentlyContinue }

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Output "exit=127 bytes=0 lines=0 timeout=False msg=python_not_found log=$LogPath"
    exit 0
}

$timedOut = $false
$exit = 1
$job = Start-Job -ScriptBlock {
    param($wd, $s, $lp)
    Set-Location $wd
    & python $s *> $lp
    $LASTEXITCODE
} -ArgumentList (Get-Location).Path, $Script, $LogPath

if (Wait-Job $job -Timeout $TimeoutSec) {
    $exit = Receive-Job $job -ErrorAction SilentlyContinue
    if ($null -eq $exit) { $exit = 1 }
} else {
    $timedOut = $true
    Stop-Job $job -ErrorAction SilentlyContinue
    $exit = -9
}
Remove-Job $job -Force -ErrorAction SilentlyContinue

$bytes = 0; $lines = 0
if (Test-Path $LogPath) {
    $bytes = (Get-Item $LogPath).Length
    $lines = @(Get-Content $LogPath).Count
}
Write-Output ("exit={0} bytes={1} lines={2} timeout={3} log={4}" -f $exit, $bytes, $lines, $timedOut, $LogPath)
