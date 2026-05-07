"""Edge tests: schedulability (Rate Monotonic) — L3."""

import math
import pytest
from conftest import load_artifact

m = load_artifact()


def _rm_bound(n: int) -> float:
    return n * (2 ** (1 / n) - 1)


def test_single_task_fully_utilised():
    # n=1 bound = 1.0; wcet=period → utilisation=1.0 → schedulable (≤ bound)
    assert m.is_schedulable([{"period": 10, "wcet": 10}]) is True


def test_single_task_exceeds_bound():
    # bound for n=1 is 1.0; wcet > period is invalid input → False
    assert m.is_schedulable([{"period": 10, "wcet": 11}]) is False


def test_two_tasks_under_bound():
    # bound n=2 ≈ 0.828; 0.5 + 0.25 = 0.75 ≤ 0.828
    tasks = [{"period": 10, "wcet": 5}, {"period": 20, "wcet": 5}]
    assert m.is_schedulable(tasks) is True


def test_two_tasks_over_bound():
    # 0.9 + 0.4 = 1.3 > 0.828
    tasks = [{"period": 10, "wcet": 9}, {"period": 10, "wcet": 4}]
    assert m.is_schedulable(tasks) is False


def test_three_tasks_classic_example():
    # Classic RM: 3 tasks that fit
    # u = 1/2 + 1/4 + 1/8 = 0.875 ≈ n*(2^(1/n)-1) for n=3 ≈ 0.780; > bound → not schedulable
    tasks = [{"period": 2, "wcet": 1}, {"period": 4, "wcet": 1}, {"period": 8, "wcet": 1}]
    # u = 0.5 + 0.25 + 0.125 = 0.875 > 0.780
    assert m.is_schedulable(tasks) is False


def test_invalid_zero_period_returns_false():
    assert m.is_schedulable([{"period": 0, "wcet": 0}]) is False


def test_invalid_negative_wcet_returns_false():
    assert m.is_schedulable([{"period": 10, "wcet": -1}]) is False


def test_empty_task_set():
    # No tasks → trivially schedulable
    result = m.is_schedulable([])
    assert result is True or result is False  # both defensible; just must not crash
