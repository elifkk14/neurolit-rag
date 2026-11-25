from app.ratelimit import SlidingWindowLimiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_allows_up_to_limit_then_reports_wait():
    clock = Clock()
    lim = SlidingWindowLimiter(2, 60, clock=clock)
    assert lim.retry_after("a") == 0
    clock.t += 10
    assert lim.retry_after("a") == 0
    clock.t += 5
    assert lim.retry_after("a") == 45  # oldest event (t=1000) leaves the window at t=1060


def test_window_slides_and_keys_are_independent():
    clock = Clock()
    lim = SlidingWindowLimiter(1, 60, clock=clock)
    assert lim.retry_after("a") == 0
    assert lim.retry_after("b") == 0
    assert lim.retry_after("a") > 0
    clock.t += 60
    assert lim.retry_after("a") == 0


def test_zero_limit_disables_and_undo_frees_a_slot():
    assert all(SlidingWindowLimiter(0, 60).retry_after("a") == 0 for _ in range(100))
    lim = SlidingWindowLimiter(1, 60, clock=Clock())
    lim.retry_after("a")
    lim.undo("a")
    assert lim.retry_after("a") == 0
