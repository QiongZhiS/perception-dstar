# vision/run_experiments.ps1 - one-line safe runners for the README quick-start
# experiments (docs/221 A5 reproducibility assets).
#
# Usage (from repo root):
#   pwsh -NoProfile -File vision\run_experiments.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File vision\run_experiments.ps1
#
# Each experiment is wrapped in exactly one line of the form
#   powershell -NoProfile -Command "& python vision\<script> *> logs\<name>.log; ..."
# so the experiment's stdout/stderr NEVER reaches the console: everything is
# redirected into logs\<name>.log and this wrapper prints ONLY one metadata
# line per experiment:  exit=<code> bytes=<log size in bytes>
# (same discipline as run_transduction_safe.ps1).
#
# RULE: never read/Get-Content the log files - metadata only.
#
# NOTE: demo_app.py starts an HTTP server and blocks until interrupted
# (Ctrl+C); its metadata line is printed only after the server stops.
# davis_suspicious.py requires DAVIS data (see davis_setup.py).
# dstar_compress.py needs vision/out/demo_storm.mp4 (its default --video);
# transduction.py generates demo_scene.mp4, NOT demo_storm.mp4, so dstar_compress
# reports exit=1 until a matching video is provided via --video (pre-existing).

$ErrorActionPreference = "Continue"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $root
$logsDir = Join-Path $root "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

powershell -NoProfile -Command "& python vision\transduction.py *> logs\transduction.log; Write-Host ('exit=' + `$LASTEXITCODE + ' bytes=' + (Get-Item 'logs\transduction.log').Length)"
powershell -NoProfile -Command "& python vision\dstar_compress.py *> logs\dstar_compress.log; Write-Host ('exit=' + `$LASTEXITCODE + ' bytes=' + (Get-Item 'logs\dstar_compress.log').Length)"
powershell -NoProfile -Command "& python vision\keep_reject.py *> logs\keep_reject.log; Write-Host ('exit=' + `$LASTEXITCODE + ' bytes=' + (Get-Item 'logs\keep_reject.log').Length)"
powershell -NoProfile -Command "& python vision\keep_reject_open.py *> logs\keep_reject_open.log; Write-Host ('exit=' + `$LASTEXITCODE + ' bytes=' + (Get-Item 'logs\keep_reject_open.log').Length)"
powershell -NoProfile -Command "& python vision\keep_reject_continuous.py *> logs\keep_reject_continuous.log; Write-Host ('exit=' + `$LASTEXITCODE + ' bytes=' + (Get-Item 'logs\keep_reject_continuous.log').Length)"
powershell -NoProfile -Command "& python vision\davis_suspicious.py *> logs\davis_suspicious.log; Write-Host ('exit=' + `$LASTEXITCODE + ' bytes=' + (Get-Item 'logs\davis_suspicious.log').Length)"
powershell -NoProfile -Command "& python vision\demo_app.py --port 8080 *> logs\demo_app.log; Write-Host ('exit=' + `$LASTEXITCODE + ' bytes=' + (Get-Item 'logs\demo_app.log').Length)"
