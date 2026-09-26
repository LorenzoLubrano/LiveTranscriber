"""Whether the keep-up monitor tells the truth about a slow machine."""

from __future__ import annotations

from app.sessions.keepup import KeepUpMonitor, Level


def feed(monitor: KeepUpMonitor, cost: float, seconds: float = 2.0, count: int = 20):
    """Simulate `count` inference passes costing `cost` per second of audio."""
    position = monitor.last_position
    for _ in range(count):
        position += seconds
        monitor.record(inference_time=cost * seconds, stream_position=position)
    return monitor


def test_no_verdict_before_there_is_evidence():
    monitor = KeepUpMonitor()
    monitor.record(inference_time=5.0, stream_position=2.0)
    # One pass says nothing: a single slow run happens on any machine.
    assert monitor.level is Level.OK
    assert monitor.load == 0.0


def test_a_fast_machine_stays_ok():
    monitor = feed(KeepUpMonitor(), cost=0.15)
    assert monitor.level is Level.OK
    assert 0.1 < monitor.load < 0.2
    assert monitor.take_warning() is None


def test_a_tight_machine_is_reported_once():
    monitor = feed(KeepUpMonitor(), cost=0.9)
    assert monitor.level is Level.TIGHT
    assert monitor.take_warning() is Level.TIGHT
    # Reporting the same level again would nag on every single pass.
    assert monitor.take_warning() is None


def test_an_overrun_is_reported_even_after_a_tight_warning():
    monitor = feed(KeepUpMonitor(), cost=0.9)
    assert monitor.take_warning() is Level.TIGHT

    feed(monitor, cost=2.0, count=40)
    assert monitor.level is Level.OVERRUN
    assert monitor.take_warning() is Level.OVERRUN
    assert monitor.take_warning() is None


def test_a_recovered_machine_is_not_warned_about_again():
    monitor = feed(KeepUpMonitor(), cost=0.9)
    assert monitor.take_warning() is Level.TIGHT

    feed(monitor, cost=0.1, count=60)
    assert monitor.level is Level.OK
    # Going back down must not re-arm the warning: the user has been told, and
    # the mitigation that fixed it is exactly what we wanted.
    feed(monitor, cost=0.9, count=40)
    assert monitor.take_warning() is None


def test_only_recent_history_counts():
    monitor = feed(KeepUpMonitor(window_seconds=30.0), cost=2.0, count=20)
    assert monitor.level is Level.OVERRUN

    # A minute of comfortable running must push the bad stretch out of view.
    feed(monitor, cost=0.1, count=40)
    assert monitor.level is Level.OK


def test_load_ignores_the_first_pass_own_cost():
    """Cost is attributed to the audio that arrived since the previous pass.

    Counting the oldest retained pass would divide a full pass of inference by
    a span that does not include the audio it processed, overstating the load.
    """
    monitor = KeepUpMonitor(min_audio_seconds=1.0)
    monitor.record(inference_time=10.0, stream_position=1.0)
    monitor.record(inference_time=1.0, stream_position=3.0)
    monitor.record(inference_time=1.0, stream_position=5.0)
    # 2 s of inference over the 4 s of audio that followed the first pass.
    assert monitor.load == 0.5


def test_a_stalled_stream_does_not_divide_by_zero():
    monitor = KeepUpMonitor(min_audio_seconds=0.0)
    monitor.record(inference_time=1.0, stream_position=4.0)
    monitor.record(inference_time=1.0, stream_position=4.0)
    assert monitor.load == 0.0
    assert monitor.level is Level.OK
