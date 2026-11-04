# Habitoo worker – video místnosti → 3D model (Gaussian splat)

Worker si z fronty `scan_jobs` (Supabase) bere nejstarší úlohu „čeká", stáhne video, vybere snímky,
spočítá polohy kamer (COLMAP), natrénuje Gaussian splat (gsplat) a nahraje **SPZ** + **PLY** do soukromého
úložiště. Pak zapíše výsledek do `scan_jobs` a smaže dočasné soubory. Jedna úloha najednou.

```
čeká → [zámek] zpracovává se → stáhnout video → 100–200 snímků (bez rozmazaných, zmenšené)
     → COLMAP polohy → gsplat trénink (strop gaussiánů, log VRAM) → SPZ + PLY → upload → hotovo
                                                     └─ chyba → nepovedlo se + důvod
```

Stav: kód a logika jsou otestované bez GPU (`pytest`, 55 testů). **Trénink na GPU a sestavení Docker
obrazu zatím nikdo nespustil** – první reálný běh je na tvém PC (kapitola 5).

## 1. Požadavky (cílový PC)

Windows 10 (21H2+), NVIDIA GTX 1660 Super (6 GB), 16 GB RAM, ovladač NVIDIA ≥ 525, WSL2, Docker Desktop
(WSL2 backend). Volné místo na disku: počítej s ~15 GB pro obraz + ~5 GB pro dočasná data.

## 2. Nastavení (jednorázově)

1. **Databáze** – v Supabase SQL editoru spusť podle pořadí:
   `sql/000_optional_scan_jobs_example.sql` (jen pokud tabulka `scan_jobs` neexistuje),
   `sql/001_scan_jobs_worker.sql`, `sql/002_video_retention.sql`.
   SQL předpokládá sloupce `id, room_id, video_path, status, created_at` – **[DOPLNIT]** ověř podle své tabulky.
2. **Buckety** v Supabase Storage: jeden pro videa, jeden pro modely. Oba nastav jako **soukromé**
   (private). Worker používá service role, žádné veřejné URL nevytváří.
3. **Klíč a nastavení** – zkopíruj `.env.example` na `.env` a doplň:
   - `SUPABASE_URL` a `SUPABASE_SERVICE_ROLE_KEY` (Supabase → Project Settings → API → `service_role`),
   - `VIDEO_BUCKET`, `MODEL_BUCKET`,
   - `JOB_TIMEOUT_S` (časový limit úlohy v sekundách) – **[DOPLNIT]**, rozhodnutí je na tobě,
   - `VIDEO_RETENTION_DAYS` – `0` = smazat video hned po úspěšném zpracování, jinak počet dní – **[DOPLNIT]**.

   Pravidla pro klíč: **nikdy ho nevkládej do kódu, do chatu/promptu, do gitu ani do Docker obrazu.**
   `.env` je v `.gitignore` i `.dockerignore`; do kontejneru se dostává jen za běhu (`env_file`).
   Pokud klíč někam unikl, vygeneruj v Supabase nový (stará hodnota přestane platit).

## 3. Kontrola GPU (před prvním spuštěním)

Rychle: `powershell -ExecutionPolicy Bypass -File scripts\check_gpu.ps1` – vypíše OK/CHYBA a co opravit.

Ručně, v tomto pořadí:

| # | Co | Příkaz (PowerShell) | Očekávání |
|---|----|---------------------|-----------|
| 1 | Ovladač NVIDIA na Windows | `nvidia-smi` | vidíš GTX 1660 Super, *Driver Version* ≥ 525 |
| 2 | WSL2 | `wsl --status`, `wsl -l -v` | výchozí verze 2, distribuce (např. Ubuntu) se `VERSION 2`; `wsl --update` |
| 3 | Docker | `docker version` | běží *Server*; v Docker Desktop zapnuto *Use the WSL 2 based engine* |
| 4 | GPU v kontejneru | `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi` | stejná tabulka jako v kroku 1 |
| 5 | torch v obrazu workeru | spouští `check_gpu.ps1` po sestavení obrazu | `torch.cuda.is_available()` = True, ~6 GB |

Ovladač instaluj **jen na Windows**, do WSL2 žádný Linuxový ovladač NVIDIA neinstaluj.

### Záložní cesta, když GPU v Dockeru nejde (krok 4 selže)

1. **Worker přímo ve WSL2 bez Dockeru** (vyžaduje, aby `nvidia-smi` fungovalo uvnitř WSL distribuce):
   v Ubuntu ve WSL2 nainstaluj CUDA toolkit 12.1 z NVIDIA repozitáře pro WSL-Ubuntu (jen toolkit, ne ovladač),
   pak `python3 -m venv .venv && . .venv/bin/activate`,
   `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`,
   `TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=2 pip install --no-build-isolation gsplat==1.5.3`, `pip install .`
   a spusť `python -m habitoo_worker.main` (načte `.env`). Postup nebyl vyzkoušen.
2. **Pronajatá cloudová GPU** – stejný obraz, stejné `.env`; je to cíl, ke kterému je worker navržený.
3. **Bez GPU nejde trénovat** (gsplat vyžaduje CUDA). Jen testovací režim:
   `python scripts/run_local.py --synthetic` ověří tok dat, kvalita výsledku nic neznamená.

## 4. Spuštění, zastavení, fronta

```powershell
scripts\run_worker.ps1        # sestaví obraz (poprvé desítky minut) a spustí worker na pozadí
docker compose logs -f worker # log (čas jednotlivých kroků, špička VRAM, chyby)
scripts\stop_worker.ps1       # zastaví worker (-Down navíc smaže kontejner)
docker compose run --rm --no-deps worker python3 scripts/queue_status.py   # počty úloh podle stavu
```

`restart: unless-stopped` – po restartu PC a spuštění Dockeru se worker spustí sám.
Alternativně fronta přímo v SQL: `select status, count(*) from scan_jobs group by status;`

## 5. První reálné video

1. Natoč 1–2 minuty pomalého záběru jedné místnosti, soubor ulož do `samples/` pod jménem
   **[DOPLNIT]** (složka je v `.gitignore`, video se nikdy necommituje).
2. Zastav běžící worker (`scripts\stop_worker.ps1`), jinak by si s testem dělily GPU.
3. Spusť (bez Supabase, falešná fronta, výsledek ve složce `out/`):

   ```powershell
   docker compose run --rm --no-deps worker python3 scripts/run_local.py --video /samples/<soubor> --mode real --out /out
   ```
4. Sleduj v logu: počet snímků, „zaregistrováno N/M snímků", `trénink … VRAM=… MB`, konečné
   `špička VRAM`. Hotovo = `out/model.spz` a `out/model.ply`.
5. Když se nevejde do paměti nebo je to pomalé, ladí se v `.env`:

   | Proměnná | Výchozí | Účinek |
   |----------|---------|--------|
   | `MAX_GAUSSIANS` | 600000 | strop počtu gaussiánů (hlavní spotřebitel VRAM) |
   | `MAX_IMAGE_SIDE` | 960 | delší strana snímků v px |
   | `TARGET_FRAMES` / `MIN_FRAMES` | 150 / 30 | kolik snímků použít / minimum |
   | `TRAIN_STEPS` | 7000 | počet kroků tréninku (čas vs. kvalita) |

   Po `CUDA out of memory` zkusí worker automaticky jednou znovu s polovičním stropem a menším
   rozlišením; pokud i to selže, úloha skončí „nedostatek paměti GPU". Hodnoty výchozích prahů
   (rozmazání, minimum zaregistrovaných snímků, počet kroků) jsou odhady – **ladí se na prvním reálném videu.**

## 6. Spolehlivost a soukromí

- **Jedna úloha najednou**; zámek je atomický v databázi (`FOR UPDATE SKIP LOCKED`).
- **Časový limit** `JOB_TIMEOUT_S`: worker úlohu sám ukončí („překročen časový limit").
- **Pád / restart PC:** úloha zůstane „zpracovává se". Worker při startu a pak každou minutu vrací úlohy
  starší než limit na „čeká"; po `MAX_ATTEMPTS` (2) pokusech je označí „nepovedlo se". Restart tedy
  znamená čekání až `JOB_TIMEOUT_S`, než se úloha vezme znovu.
- **Důvody selhání** (sloupec `failure_reason`): málo záběrů · rozmazané video · nepodařilo se spočítat
  polohy · překročen časový limit · nedostatek paměti GPU · video se nepodařilo stáhnout · jiná chyba.
- **Soukromí:** buckety jsou soukromé; worker nesbírá žádná další data, jen zapisuje výsledek do `scan_jobs`;
  původní video se maže podle `VIDEO_RETENTION_DAYS` (videa selhaných úloh stejným pravidlem);
  pracovní složka úlohy se maže vždy, i po chybě. Logy neobsahují klíč.
- Výstupy leží v bucketu modelů jako `<room_id>/<job_id>.spz` a `.ply`. **[DOPLNIT]** Supabase má limit
  velikosti souboru podle plánu – PLY (~140 MB při 600 tis. gaussiánech) ho může překročit.

## 7. Vývoj a testy

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]" && pytest                # ~25 s, bez GPU
python scripts/run_local.py --synthetic --out out   # celá pipeline se syntetickými náhradami
```

Testy kryjí frontu, stavy, requeue, konfiguraci, snímky, export, runner a **skutečný COLMAP** (CPU)
na syntetické scéně. **Nekryjí:** trénink gsplat na GPU, Docker build, PowerShell skripty, skutečný Supabase.

## 8. Struktura

`src/habitoo_worker/` – `main.py` (smyčka), `config.py`, `jobs/` (fronta: Supabase + falešná),
`storage.py`, `pipeline/` (`frames`, `poses`, `train`, `export`, `runner`, `synthetic/`) ·
`sql/` migrace · `scripts/` (PowerShell + `run_local.py`, `queue_status.py`) · `tests/` · `Dockerfile`,
`docker-compose.yml` · licence knihoven: [LICENSES.md](LICENSES.md)

## 9. Otevřené body

[DOPLNIT]: schéma `scan_jobs` a názvy bucketů · `JOB_TIMEOUT_S` · `VIDEO_RETENTION_DAYS` · jméno vzorového
souboru · limit velikosti souboru v úložišti · architektura cloudové GPU (`--build-arg CUDA_ARCHS`).
Neověřeno bez GPU: trénink a jeho paměť, shoda SPZ s oficiálním loaderem Niantic (ověří prohlížeč modelu),
Docker build, PowerShell skripty.
