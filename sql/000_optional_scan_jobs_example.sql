-- VOLITELNÉ: spusť jen pokud tabulka scan_jobs ještě NEEXISTUJE.
-- Je to minimální návrh; sloupce označené [DOPLNIT] přizpůsob své appce.
create table if not exists public.scan_jobs (
  id          uuid primary key default gen_random_uuid(),
  room_id     uuid not null,            -- [DOPLNIT] odkaz na tabulku místností
  video_path  text not null,            -- cesta k videu v bucketu s videi
  status      text not null default 'čeká'
              check (status in ('čeká','zpracovává se','hotovo','nepovedlo se')),
  created_at  timestamptz not null default now()
);
alter table public.scan_jobs enable row level security;
-- [DOPLNIT] RLS politiky pro appku (worker používá service role a RLS obchází).
