"""Vypíše počty úloh podle stavu a posledních 10 úloh. Čte SUPABASE_URL a klíč z .env / env."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

from habitoo_worker.config import Config  # noqa: E402
from habitoo_worker.jobs.supabase_store import SupabaseJobStore  # noqa: E402


def main() -> int:
    load_dotenv()
    cfg = Config.from_env(os.environ)
    client = create_client(cfg.supabase_url, cfg.service_role_key)
    for status, n in SupabaseJobStore(client, cfg.jobs_table).status_counts().items():
        print(f"{status:15} {n}")
    rows = (client.table(cfg.jobs_table)
            .select("id, status, attempts, claimed_at, failure_reason")
            .order("created_at", desc=True).limit(10).execute().data or [])
    print("\nposledních 10 úloh:")
    for r in rows:
        print(f"  {r['id']}  {r['status']:15} pokusů={r.get('attempts')}  {r.get('failure_reason') or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
