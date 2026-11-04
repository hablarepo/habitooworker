<#
Zastaví worker (SIGTERM, počká až 30 s). Rozpracovaná úloha zůstane ve stavu "zpracovává se"
a po uplynutí JOB_TIMEOUT_S ji worker při dalším startu vrátí do fronty.
  powershell -ExecutionPolicy Bypass -File scripts\stop_worker.ps1 [-Down]
-Down navíc odstraní kontejner (obraz i data zůstanou).
#>
param([switch]$Down)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if ($Down) { docker compose down } else { docker compose stop }
