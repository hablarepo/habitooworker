import pytest

from habitoo_worker.jobs.fake_store import FakeJobStore
from habitoo_worker.jobs.models import (
    FAIL_BLURRY, FAIL_TIMEOUT, STATUS_DONE, STATUS_FAILED, STATUS_PENDING,
    STATUS_PROCESSING, JobResult, JobStateError,
)

RESULT = JobResult("a.spz", "a.ply", 100, 400, 5000, 12.3)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(clock):
    return FakeJobStore(clock=clock)


def test_claim_returns_oldest_pending(store):
    first = store.add_job()
    store.add_job()
    job = store.claim_next()
    assert job.id == first
    assert store.get(first).status == STATUS_PROCESSING
    assert job.attempts == 1


def test_empty_queue_returns_none(store):
    assert store.claim_next() is None


def test_two_claims_never_return_same_job(store):
    a, b = store.add_job(), store.add_job()
    ids = {store.claim_next().id, store.claim_next().id}
    assert ids == {a, b}
    assert store.claim_next() is None


def test_claim_skips_non_pending(store):
    a = store.add_job()
    b = store.add_job()
    store.claim_next()
    store.mark_failed(a, FAIL_BLURRY)
    assert store.claim_next().id == b


def test_mark_done_stores_result(store):
    jid = store.add_job()
    store.claim_next()
    store.mark_done(jid, RESULT)
    row = store.get(jid)
    assert row.status == STATUS_DONE and row.result == RESULT


def test_mark_failed_stores_reason(store):
    jid = store.add_job()
    store.claim_next()
    store.mark_failed(jid, FAIL_BLURRY, "detail")
    row = store.get(jid)
    assert row.status == STATUS_FAILED
    assert row.failure_reason == FAIL_BLURRY and row.failure_detail == "detail"


def test_mark_requires_processing_state(store):
    jid = store.add_job()  # ještě "čeká"
    with pytest.raises(JobStateError):
        store.mark_done(jid, RESULT)
    with pytest.raises(JobStateError):
        store.mark_failed(jid, FAIL_BLURRY)


def test_fresh_processing_job_is_not_requeued(store, clock):
    jid = store.add_job()
    store.claim_next()
    clock.t += 50
    report = store.requeue_stale(timeout_s=100, max_attempts=2)
    assert (report.requeued, report.failed) == (0, 0)
    assert store.get(jid).status == STATUS_PROCESSING


def test_stale_job_goes_back_to_pending_then_fails_after_max_attempts(store, clock):
    jid = store.add_job()
    store.claim_next()                       # pokus 1
    clock.t += 101
    report = store.requeue_stale(timeout_s=100, max_attempts=2)
    assert (report.requeued, report.failed) == (1, 0)
    assert store.get(jid).status == STATUS_PENDING

    assert store.claim_next().attempts == 2  # pokus 2
    clock.t += 101
    report = store.requeue_stale(timeout_s=100, max_attempts=2)
    assert (report.requeued, report.failed) == (0, 1)
    row = store.get(jid)
    assert row.status == STATUS_FAILED and row.failure_reason == FAIL_TIMEOUT


def test_requeued_job_cannot_be_finished_by_old_worker(store, clock):
    jid = store.add_job()
    store.claim_next()
    clock.t += 101
    store.requeue_stale(timeout_s=100, max_attempts=2)
    with pytest.raises(JobStateError):
        store.mark_done(jid, RESULT)


def test_status_counts(store):
    store.add_job()
    store.add_job()
    store.claim_next()
    counts = store.status_counts()
    assert counts[STATUS_PENDING] == 1 and counts[STATUS_PROCESSING] == 1
    assert counts[STATUS_DONE] == 0 and counts[STATUS_FAILED] == 0
