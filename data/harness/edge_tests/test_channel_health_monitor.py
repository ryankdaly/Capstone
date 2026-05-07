"""Edge tests: channel_health_monitor — L4."""

import pytest
from conftest import load_artifact

m = load_artifact()


def make_mon(**kwargs):
    defaults = dict(
        n_channels=2,
        sample_window=5,
        range_lo=0,
        range_hi=100,
        max_rate=10,
        stuck_threshold=3,
    )
    defaults.update(kwargs)
    return m.ChannelHealthMonitor(**defaults)


def status_name(s) -> str:
    return s.name if hasattr(s, "name") else str(s)


def test_ok_status():
    mon = make_mon()
    status = mon.ingest(0, 50, 0)
    assert status_name(status) == "OK"


def test_out_of_range_high():
    mon = make_mon()
    status = mon.ingest(0, 101, 0)
    assert status_name(status) == "OUT_OF_RANGE"


def test_out_of_range_low():
    mon = make_mon()
    status = mon.ingest(0, -1, 0)
    assert status_name(status) == "OUT_OF_RANGE"


def test_rate_violation():
    mon = make_mon()
    mon.ingest(0, 50, 0)
    status = mon.ingest(0, 61, 100)  # delta=11 > max_rate=10
    assert status_name(status) == "RATE_VIOLATION"


def test_stuck_fault():
    mon = make_mon()
    for t in range(3):
        mon.ingest(0, 50, t * 100)
    status = mon.ingest(0, 50, 300)  # 4th identical → stuck_threshold=3 exceeded
    assert status_name(status) == "STUCK"


def test_intermittent():
    mon = make_mon(sample_window=5)
    mon.ingest(0, 50, 0)    # OK
    mon.ingest(0, 200, 100) # OUT_OF_RANGE (fault)
    mon.ingest(0, 50, 200)  # OK (recovery)
    mon.ingest(0, 50, 300)  # OK
    status = mon.ingest(0, 50, 400)  # still in window with a fault → INTERMITTENT
    assert status_name(status) == "INTERMITTENT"


def test_out_of_range_priority_over_rate_violation():
    mon = make_mon()
    mon.ingest(0, 50, 0)
    # delta=60 > max_rate=10 AND value=110 > range_hi=100
    # OUT_OF_RANGE should win (checked first)
    status = mon.ingest(0, 110, 100)
    assert status_name(status) == "OUT_OF_RANGE"


def test_data_loss():
    mon = make_mon(sample_window=10)
    # Establish median interval ≈ 100ms with 3 samples
    mon.ingest(0, 50, 0)
    mon.ingest(0, 50, 100)
    mon.ingest(0, 50, 200)
    # 2 × median = 200ms; gap of 500ms → DATA_LOSS
    status = mon.ingest(0, 50, 700)
    assert status_name(status) == "DATA_LOSS"


def test_invalid_channel_id_raises():
    mon = make_mon(n_channels=2)
    with pytest.raises((ValueError, IndexError, KeyError)):
        mon.ingest(5, 50, 0)  # channel_id=5 >= n_channels=2


def test_invalid_construction_range_lo_gt_hi():
    with pytest.raises(ValueError):
        m.ChannelHealthMonitor(
            n_channels=1, sample_window=5,
            range_lo=100, range_hi=0,
            max_rate=10, stuck_threshold=3,
        )


def test_channel_report_structure():
    mon = make_mon()
    mon.ingest(0, 50, 0)
    report = mon.get_channel_report(0)
    assert "current_status" in report
    assert "fault_count_in_window" in report
    assert "last_good_value" in report
    assert "last_good_timestamp_ms" in report
