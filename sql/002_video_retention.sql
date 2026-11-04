-- Sloupec pro evidenci smazaných původních videí (retence, viz VIDEO_RETENTION_DAYS).
alter table public.scan_jobs
  add column if not exists video_deleted_at timestamptz;

create index if not exists scan_jobs_video_purge_idx
  on public.scan_jobs (finished_at)
  where video_deleted_at is null and status in ('hotovo', 'nepovedlo se');
