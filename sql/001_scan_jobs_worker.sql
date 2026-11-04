-- Migrace pro worker. Spusť ručně v Supabase SQL editoru. Worker schéma sám nemění.
-- [DOPLNIT] Předpokládá existující sloupce: id, room_id, video_path, status, created_at.
-- Řetězce stavů musí odpovídat src/habitoo_worker/jobs/models.py.

alter table public.scan_jobs
  add column if not exists attempts           integer     not null default 0,
  add column if not exists claimed_at         timestamptz,
  add column if not exists finished_at        timestamptz,
  add column if not exists failure_reason     text,
  add column if not exists failure_detail     text,
  add column if not exists result_spz_path    text,
  add column if not exists result_ply_path    text,
  add column if not exists result_spz_bytes   bigint,
  add column if not exists result_ply_bytes   bigint,
  add column if not exists gaussian_count     integer,
  add column if not exists processing_seconds real;

create index if not exists scan_jobs_status_created_idx
  on public.scan_jobs (status, created_at);

-- Atomicky vezme nejstarší úlohu "čeká". SKIP LOCKED zaručí, že dva workery nevezmou tutéž.
create or replace function public.claim_next_scan_job()
returns setof public.scan_jobs
language sql
as $$
  update public.scan_jobs j
     set status = 'zpracovává se',
         attempts = j.attempts + 1,
         claimed_at = now(),
         failure_reason = null,
         failure_detail = null
   where j.id = (
     select id from public.scan_jobs
      where status = 'čeká'
      order by created_at asc
      limit 1
      for update skip locked
   )
  returning j.*;
$$;

-- Zaseknuté úlohy (worker spadl / PC restart): attempts < max -> "čeká", jinak "nepovedlo se".
create or replace function public.requeue_stale_scan_jobs(p_timeout_seconds integer, p_max_attempts integer)
returns table (requeued integer, failed integer)
language plpgsql
as $$
declare
  v_requeued integer;
  v_failed integer;
begin
  with failed_rows as (
    update public.scan_jobs
       set status = 'nepovedlo se',
           failure_reason = 'překročen časový limit',
           finished_at = now()
     where status = 'zpracovává se'
       and claimed_at < now() - make_interval(secs => p_timeout_seconds)
       and attempts >= p_max_attempts
    returning 1
  )
  select count(*) into v_failed from failed_rows;

  with requeued_rows as (
    update public.scan_jobs
       set status = 'čeká', claimed_at = null
     where status = 'zpracovává se'
       and claimed_at < now() - make_interval(secs => p_timeout_seconds)
       and attempts < p_max_attempts
    returning 1
  )
  select count(*) into v_requeued from requeued_rows;

  return query select v_requeued, v_failed;
end;
$$;

-- Funkce smí volat jen service role (worker), ne anon/přihlášení uživatelé appky.
revoke all on function public.claim_next_scan_job() from public, anon, authenticated;
revoke all on function public.requeue_stale_scan_jobs(integer, integer) from public, anon, authenticated;
grant execute on function public.claim_next_scan_job() to service_role;
grant execute on function public.requeue_stale_scan_jobs(integer, integer) to service_role;
