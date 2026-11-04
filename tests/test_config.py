import pytest

from habitoo_worker.config import Config, ConfigError

BASE = {
    "SUPABASE_URL": "https://x.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "secret-key",
    "VIDEO_BUCKET": "videos", "MODEL_BUCKET": "models",
    "JOB_TIMEOUT_S": "3600", "VIDEO_RETENTION_DAYS": "0",
}


def test_valid_config_and_defaults():
    cfg = Config.from_env(BASE)
    assert cfg.max_attempts == 2 and cfg.jobs_table == "scan_jobs"


def test_key_not_in_repr():
    assert "secret-key" not in repr(Config.from_env(BASE))


@pytest.mark.parametrize("missing", ["JOB_TIMEOUT_S", "VIDEO_RETENTION_DAYS", "VIDEO_BUCKET",
                                     "MODEL_BUCKET", "SUPABASE_SERVICE_ROLE_KEY"])
def test_real_mode_requires_doplnit_values(missing):
    env = {k: v for k, v in BASE.items() if k != missing}
    with pytest.raises(ConfigError, match=missing):
        Config.from_env(env)


def test_synthetic_mode_needs_nothing():
    assert Config.from_env({"WORKER_MODE": "synthetic"}).mode == "synthetic"


def test_invalid_values():
    with pytest.raises(ConfigError):
        Config.from_env({**BASE, "JOB_TIMEOUT_S": "abc"})
    with pytest.raises(ConfigError):
        Config.from_env({**BASE, "MIN_FRAMES": "500"})
    with pytest.raises(ConfigError):
        Config.from_env({**BASE, "WORKER_MODE": "gpu"})


def test_train_steps_default_and_override():
    assert Config.from_env(BASE).train_steps == 7000
    assert Config.from_env({**BASE, "TRAIN_STEPS": "100"}).train_steps == 100
    with pytest.raises(ConfigError):
        Config.from_env({**BASE, "TRAIN_STEPS": "0"})
