<#
Kontrola, že GPU půjde použít v Dockeru přes WSL2. Spusť v PowerShellu na Windows:
  powershell -ExecutionPolicy Bypass -File scripts\check_gpu.ps1
Nic nemění, jen vypisuje OK / CHYBA a co s tím.
#>
$ErrorActionPreference = "Continue"
$script:fail = 0
$CudaImage = "nvidia/cuda:12.1.1-base-ubuntu22.04"
$MinDriver = 525   # CUDA 12.1 v kontejneru potřebuje ovladač NVIDIA >= 525

function Ok($msg)  { Write-Host "[ OK ]  $msg" -ForegroundColor Green }
function Bad($msg, $hint) {
    Write-Host "[CHYBA] $msg" -ForegroundColor Red
    if ($hint) { Write-Host "        -> $hint" -ForegroundColor Yellow }
    $script:fail++
}

Write-Host "== 1. Windows =="
$build = [Environment]::OSVersion.Version.Build
Write-Host "Build Windows: $build"
if ($build -ge 19044) { Ok "Windows podporuje GPU ve WSL2 (build >= 19044, tj. 10 21H2+)" }
else { Bad "Starý build Windows ($build)" "Aktualizuj Windows 10 na verzi 21H2 nebo novější (Windows Update)." }

Write-Host "`n== 2. Ovladač NVIDIA na Windows =="
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Bad "nvidia-smi nenalezeno" "Nainstaluj aktuální ovladač NVIDIA pro GTX 1660 Super z nvidia.com (Windows ovladač, ne Linux)."
} else {
    $line = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) 2>&1 | Select-Object -First 1
    Write-Host "GPU: $line"
    $parts = $line -split ","
    $ver = [double](($parts[1].Trim()) -replace ",", ".")
    if ($ver -ge $MinDriver) { Ok "Ovladač $ver (minimum $MinDriver)" }
    else { Bad "Ovladač $ver je starý" "Aktualizuj ovladač NVIDIA na verzi $MinDriver nebo novější." }
}

Write-Host "`n== 3. WSL2 =="
$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if (-not $wsl) { Bad "wsl.exe nenalezeno" "Zapni WSL: PowerShell jako správce -> wsl --install" }
else {
    $list = ((wsl -l -v) 2>&1 | Out-String) -replace "`0", ""
    Write-Host $list
    if ($list -match "\s2\s*$" -or $list -match "\s2\r?\n") { Ok "Existuje distribuce běžící ve WSL verze 2" }
    else { Bad "Žádná distribuce ve WSL2" "wsl --set-default-version 2 ; wsl --install -d Ubuntu ; případně wsl --update" }
}

Write-Host "`n== 4. Docker =="
$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) { Bad "docker nenalezen" "Nainstaluj Docker Desktop a v nastavení zapni 'Use the WSL 2 based engine'." }
else {
    $sv = (& docker version --format "{{.Server.Version}}") 2>&1
    if ($LASTEXITCODE -eq 0) { Ok "Docker daemon běží (verze $sv)" }
    else { Bad "Docker daemon neběží" "Spusť Docker Desktop a počkej, až bude 'running'." }
}

Write-Host "`n== 5. GPU uvnitř kontejneru (nvidia-smi) =="
if ($docker -and $script:fail -eq 0) {
    $out = (& docker run --rm --gpus all $CudaImage nvidia-smi) 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0 -and $out -match "NVIDIA-SMI") { Ok "nvidia-smi v kontejneru funguje"; Write-Host $out }
    else {
        Write-Host $out
        Bad "GPU v kontejneru nefunguje" "Viz README, kapitola 'Záložní cesta, když GPU v Dockeru nejde'."
    }
} else { Write-Host "(přeskočeno – nejdřív oprav předchozí chyby)" }

Write-Host "`n== 6. PyTorch + gsplat v obrazu workeru (jen pokud je obraz sestavený) =="
if ($docker) {
    & docker image inspect habitoo-worker:local *> $null
    if ($LASTEXITCODE -eq 0) {
        $py = "import torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print(p.name, round(p.total_memory/2**30,1), 'GB, CUDA', torch.version.cuda); import gsplat; print('gsplat', gsplat.__version__)"
        $o = (& docker run --rm --gpus all habitoo-worker:local python3 -c $py) 2>&1 | Out-String
        if ($LASTEXITCODE -eq 0) { Ok "torch vidí GPU"; Write-Host $o }
        else { Write-Host $o; Bad "torch v obraze GPU nevidí" "Zkontroluj kroky 2-5; poslední řádky výpisu výše pošli dál." }
    } else { Write-Host "(obraz habitoo-worker:local zatím není, sestaví ho scripts\run_worker.ps1)" }
}

Write-Host ""
if ($script:fail -eq 0) { Write-Host "VŠE V POŘÁDKU" -ForegroundColor Green; exit 0 }
else { Write-Host "Počet chyb: $($script:fail)" -ForegroundColor Red; exit 1 }
