"""SupabaseJobStore proti atrapě klienta – ověřuje mapování volání, ne skutečnou databázi."""
import pytest

from habitoo_worker.jobs.models import STATUS_DONE, STATUS_PROCESSING, JobResult, JobStateError
from habitoo_worker.jobs.supabase_store import SupabaseJobStore


class Resp:
    def __init__(self, data=None, count=None):
        self.data, self.count = data, count


class Query:
    def __init__(self, client, name=None):
        self.client, self.name, self.calls = client, name, []

    def __getattr__(self, attr):
        def call(*a, **kw):
            self.calls.append((attr, a, kw))
            return self
        return call

    def execute(self):
        self.client.executed.append((self.name, self.calls))
        return self.client.responses.pop(0)


class Client:
    def __init__(self, responses):
        self.responses, self.executed = list(responses), []

    def rpc(self, name, params=None):
        return Query(self, f"rpc:{name}:{params}")

    def table(self, name):
        return Query(self, f"table:{name}")


def test_claim_maps_row_to_job():
    row = {"id": "j1", "room_id": "r1", "video_path": "x/v.mp4", "attempts": 1}
    store = SupabaseJobStore(Client([Resp([row])]))
    job = store.claim_next()
    assert (job.id, job.room_id, job.video_path, job.attempts) == ("j1", "r1", "x/v.mp4", 1)


@pytest.mark.parametrize("data", [[], None, [{"id": None}]])
def test_claim_empty(data):
    assert SupabaseJobStore(Client([Resp(data)])).claim_next() is None


def test_requeue_passes_params_and_parses():
    client = Client([Resp([{"requeued": 2, "failed": 1}])])
    report = SupabaseJobStore(client).requeue_stale(600, 2)
    assert (report.requeued, report.failed) == (2, 1)
    assert "'p_timeout_seconds': 600" in client.executed[0][0]


def test_mark_done_guards_on_processing_state():
    client = Client([Resp([{"id": "j1"}])])
    SupabaseJobStore(client).mark_done("j1", JobResult("a.spz", "a.ply", 1, 2, 3, 4.0))
    calls = client.executed[0][1]
    update = next(c for c in calls if c[0] == "update")[1][0]
    assert update["status"] == STATUS_DONE and update["gaussian_count"] == 3
    eqs = [c[1] for c in calls if c[0] == "eq"]
    assert ("status", STATUS_PROCESSING) in eqs and ("id", "j1") in eqs


def test_mark_failed_raises_when_job_no_longer_ours():
    store = SupabaseJobStore(Client([Resp([])]))
    with pytest.raises(JobStateError):
        store.mark_failed("j1", "důvod")
