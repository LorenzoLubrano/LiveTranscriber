"""Ring buffer: bounded memory, correct ordering, honest overflow accounting."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from app.audio.ring_buffer import RingBuffer


def test_write_then_read_roundtrip():
    rb = RingBuffer(100)
    data = np.arange(10, dtype=np.float32)
    assert rb.write(data) == 0
    assert rb.available() == 10
    np.testing.assert_array_equal(rb.read(10), data)
    assert rb.available() == 0


def test_read_returns_oldest_first():
    rb = RingBuffer(100)
    rb.write(np.array([1, 2, 3], dtype=np.float32))
    rb.write(np.array([4, 5, 6], dtype=np.float32))
    np.testing.assert_array_equal(rb.read(6), np.array([1, 2, 3, 4, 5, 6], np.float32))


def test_partial_read_leaves_remainder():
    rb = RingBuffer(100)
    rb.write(np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(rb.read(4), np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(rb.read(100), np.arange(4, 10, dtype=np.float32))


def test_read_more_than_available_returns_what_exists():
    rb = RingBuffer(100)
    rb.write(np.ones(5, dtype=np.float32))
    assert rb.read(50).size == 5
    assert rb.read(50).size == 0


def test_wraparound_preserves_order():
    rb = RingBuffer(10)
    rb.write(np.arange(8, dtype=np.float32))
    rb.read(6)                                   # read pos now 6
    rb.write(np.arange(100, 106, dtype=np.float32))  # wraps past the end
    np.testing.assert_array_equal(
        rb.read(8),
        np.array([6, 7, 100, 101, 102, 103, 104, 105], dtype=np.float32),
    )


def test_overflow_drops_oldest_and_counts():
    rb = RingBuffer(10)
    rb.write(np.arange(10, dtype=np.float32))
    dropped = rb.write(np.arange(100, 105, dtype=np.float32))

    assert dropped == 5
    assert rb.dropped_frames == 5
    assert rb.available() == 10, "capacity must never be exceeded"
    # The five oldest are gone; the newest five are intact.
    np.testing.assert_array_equal(
        rb.read(10),
        np.array([5, 6, 7, 8, 9, 100, 101, 102, 103, 104], dtype=np.float32),
    )


def test_write_larger_than_capacity_keeps_newest_tail():
    rb = RingBuffer(5)
    dropped = rb.write(np.arange(12, dtype=np.float32))

    assert rb.available() == 5
    assert dropped == 7
    np.testing.assert_array_equal(rb.read(5), np.arange(7, 12, dtype=np.float32))


def test_memory_is_bounded_over_a_long_run():
    """The property that makes multi-hour recording safe."""
    rb = RingBuffer(1000)
    chunk = np.ones(997, dtype=np.float32)  # deliberately coprime with capacity
    for _ in range(5000):
        rb.write(chunk)
    assert rb._buf.size == 1000, "backing array must never grow"
    assert rb.available() <= 1000


def test_multichannel_frames_stay_intact():
    rb = RingBuffer(10, channels=2)
    stereo = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    rb.write(stereo)
    assert rb.available() == 3
    np.testing.assert_array_equal(rb.read(3), stereo)


def test_multichannel_partial_frame_is_not_written():
    rb = RingBuffer(10, channels=2)
    rb.write(np.array([1, 2, 3], dtype=np.float32))  # 1.5 frames
    assert rb.available() == 1
    np.testing.assert_array_equal(rb.read(1), np.array([[1, 2]], dtype=np.float32))


def test_peek_does_not_consume():
    rb = RingBuffer(10)
    rb.write(np.arange(5, dtype=np.float32))
    np.testing.assert_array_equal(rb.peek(3), np.arange(3, dtype=np.float32))
    assert rb.available() == 5


def test_discard_and_clear():
    rb = RingBuffer(10)
    rb.write(np.arange(8, dtype=np.float32))
    assert rb.discard(3) == 3
    assert rb.available() == 5
    rb.clear()
    assert rb.available() == 0


def test_reset_zeroes_counters():
    rb = RingBuffer(4)
    rb.write(np.arange(10, dtype=np.float32))
    assert rb.dropped_frames > 0
    rb.reset()
    assert rb.dropped_frames == 0
    assert rb.available() == 0
    assert rb.total_written == 0


def test_int16_input_is_converted_to_float32():
    rb = RingBuffer(10)
    rb.write(np.array([1, 2, 3], dtype=np.int16))
    assert rb.read(3).dtype == np.float32


def test_empty_write_is_a_noop():
    rb = RingBuffer(10)
    assert rb.write(np.zeros(0, dtype=np.float32)) == 0
    assert rb.available() == 0


@pytest.mark.parametrize("capacity,channels", [(0, 1), (-1, 1), (10, 0)])
def test_invalid_construction_rejected(capacity, channels):
    with pytest.raises(ValueError):
        RingBuffer(capacity, channels)


def test_concurrent_producer_consumer_loses_nothing_unexpected():
    """Mimics the real topology: audio thread writes, pipeline thread reads."""
    rb = RingBuffer(8192)
    chunk = np.ones(256, dtype=np.float32)
    writes = 2000
    read_total = 0
    stop = threading.Event()

    def producer():
        for _ in range(writes):
            rb.write(chunk)
        stop.set()

    def consumer():
        nonlocal read_total
        while not stop.is_set() or rb.available():
            read_total += rb.read(512).size

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not t1.is_alive() and not t2.is_alive()
    assert read_total + rb.dropped_frames == writes * 256
