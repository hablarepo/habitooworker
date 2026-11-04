<#
Sestaví obraz (poprvé trvá desítky minut) a spustí worker na pozadí.
  powershell -ExecutionPolicy Bypass -File scripts\run_worker.ps1
Hodnoty z .env se nikdy nevypisují.
#>
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".env")) {
    Write-Host "Chybí soubor .env. Zkopíruj .env.example na .env a doplň hodnoty (viz README)." -ForegroundColor Red
    exit 1
}

$required = "SUPABASE_URL","SUPABASE_SERVICE_ROLE_KEY","JOB_TIMEOUT_S","VIDEO_RETENTION_DAYS","VIDEO_BUCKET","MODEL_BUCKET"
$values = @{}
Get-Content ".env" -Encoding UTF8 | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') { $values[$Matches[1]] = $Matches[2].Trim() }
}
$missing = $required | Where-Object { -not $values.ContainsKey($_) -or [string]::IsNullOrWhiteSpace($values[$_]) }
if ($missing) {
    Write-Host ("V .env chybí hodnoty: " + ($missing -join ", ")) -ForegroundColor Red
    exit 1
}

docker compose up -d --build
if ($LASTEXITCODE -ne 0) { Write-Host "docker compose selhal" -ForegroundColor Red; exit 1 }

Write-Host "`nWorker běží. Log:     docker compose logs -f worker"
Write-Host "Zastavení:             scripts\stop_worker.ps1"
Write-Host "Fronta úloh:           docker compose run --rm --no-deps worker python3 scripts/queue_status.py"
