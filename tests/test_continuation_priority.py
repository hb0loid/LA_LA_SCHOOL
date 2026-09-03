from __future__ import annotations

import heapq
import unittest

from laladub.bot import (
    CONTINUATION_PRIORITY_BOOST,
    PREMIUM_PRIORITY,
    _continuation_priority,
    _QueuedJob,
)


def _item(key: str, priority: int, sequence: int) -> _QueuedJob:
    return _QueuedJob(
        key=key,
        priority=priority,
        sequence=sequence,
        chat_id=1,
        user_id=1,
        job={"job_dir": f"runs/{key}", "input_path": f"runs/{key}/input.mp4"},
        status_message=None,
        enqueued_at=0.0,
        premium=priority == 0,
        queue_limit=None,
    )


def _boost(item: _QueuedJob) -> None:
    """The requeue path applied to a job coming back from the worker."""
    if not item.job.get("continuation_boosted"):
        item.priority = _continuation_priority(item.priority, item.premium)
        item.job["continuation_boosted"] = True


def _order(items: list[_QueuedJob]) -> list[str]:
    heap: list[tuple[int, int, _QueuedJob]] = []
    for item in items:
        heapq.heappush(heap, (item.priority, item.sequence, item))
    return [heapq.heappop(heap)[2].key for _ in range(len(heap))]


class ContinuationPriorityTests(unittest.TestCase):
    def test_a_returning_job_goes_before_a_later_arrival(self) -> None:
        # The point of the boost: half-finished work should not wait behind
        # work that has not started.
        arrival = _item("fresh", priority=100, sequence=2)
        returning = _item("returning", priority=100, sequence=1)
        _boost(returning)
        self.assertEqual(_order([arrival, returning]), ["returning", "fresh"])

    def test_a_returning_job_goes_before_an_earlier_arrival_too(self) -> None:
        arrival = _item("fresh", priority=100, sequence=1)
        returning = _item("returning", priority=100, sequence=2)
        _boost(returning)
        self.assertEqual(_order([arrival, returning]), ["returning", "fresh"])

    def test_a_premium_arrival_still_beats_an_ordinary_continuation(self) -> None:
        # The boost reorders within a tier; it must not let an ordinary job
        # overtake a premium one.
        premium = _item("premium", priority=0, sequence=9)
        returning = _item("returning", priority=100, sequence=1)
        _boost(returning)
        self.assertEqual(_order([premium, returning]), ["premium", "returning"])

    def test_a_premium_continuation_beats_a_premium_arrival(self) -> None:
        premium = _item("premium", priority=0, sequence=1)
        returning = _item("returning", priority=0, sequence=2)
        _boost(returning)
        self.assertEqual(_order([premium, returning]), ["returning", "premium"])

    def test_two_continuations_keep_their_original_order(self) -> None:
        first = _item("first", priority=100, sequence=1)
        second = _item("second", priority=100, sequence=2)
        _boost(first)
        _boost(second)
        self.assertEqual(_order([second, first]), ["first", "second"])

    def test_the_boost_is_applied_only_once(self) -> None:
        # A job can come back from the worker more than once (a retry, or the
        # offline fallback); it must not sink the whole queue by compounding.
        returning = _item("returning", priority=100, sequence=1)
        _boost(returning)
        after_first = returning.priority
        _boost(returning)
        _boost(returning)
        self.assertEqual(returning.priority, after_first)


    def test_no_ordinary_priority_can_ever_reach_premium(self) -> None:
        # Paying users always go first. This must hold structurally, not
        # because the boost happens to be smaller than the tier gap - a bigger
        # karma bonus or a bigger boost must not change it.
        for priority in range(1, 400):
            with self.subTest(priority=priority):
                self.assertGreater(_continuation_priority(priority, premium=False), PREMIUM_PRIORITY)

    def test_a_premium_continuation_is_not_floored(self) -> None:
        self.assertEqual(
            _continuation_priority(PREMIUM_PRIORITY, premium=True),
            PREMIUM_PRIORITY - CONTINUATION_PRIORITY_BOOST,
        )

    def test_an_ordinary_continuation_still_beats_an_ordinary_arrival(self) -> None:
        # The floor must not flatten the tier: continuations still go first
        # among ordinary jobs.
        self.assertLess(_continuation_priority(100, premium=False), 100)


if __name__ == "__main__":
    unittest.main()
