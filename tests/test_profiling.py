from __future__ import annotations


def test_profile_call_returns_value_time_and_peak_memory() -> None:
    from tp2.profiling import profile_call

    result = profile_call(lambda value: [value] * 3, 7)

    assert result.value == [7, 7, 7]
    assert result.elapsed_seconds >= 0
    assert result.peak_bytes >= 0


def test_profile_call_preserves_an_outer_tracemalloc_session() -> None:
    import tracemalloc

    from tp2.profiling import profile_call

    tracemalloc.start()
    try:
        result = profile_call(lambda: [0] * 100_000)
        _, outer_peak = tracemalloc.get_traced_memory()
        assert result.peak_bytes > 0
        assert outer_peak > 0
    finally:
        tracemalloc.stop()
