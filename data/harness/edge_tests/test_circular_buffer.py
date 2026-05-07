"""Edge tests: circular_buffer — L2."""

import pytest
from conftest import load_artifact

m = load_artifact()


def make_buf(cap: int):
    return m.CircularBuffer(cap)


def test_empty_pop_returns_none():
    buf = make_buf(3)
    assert buf.pop() is None


def test_push_and_pop_fifo_order():
    buf = make_buf(3)
    buf.push(1)
    buf.push(2)
    buf.push(3)
    assert buf.pop() == 1
    assert buf.pop() == 2
    assert buf.pop() == 3


def test_overrun_overwrites_oldest():
    buf = make_buf(3)
    buf.push(1)
    buf.push(2)
    buf.push(3)
    buf.push(4)  # overwrites 1
    assert buf.pop() == 2


def test_capacity_never_exceeded():
    cap = 4
    buf = make_buf(cap)
    for i in range(10):
        buf.push(i)
    popped = []
    while not buf.is_empty:
        popped.append(buf.pop())
    assert len(popped) == cap


def test_is_full_and_is_empty_properties():
    buf = make_buf(2)
    assert buf.is_empty is True
    assert buf.is_full is False
    buf.push("a")
    buf.push("b")
    assert buf.is_full is True
    assert buf.is_empty is False


def test_capacity_one_overrun():
    buf = make_buf(1)
    buf.push(10)
    buf.push(20)  # overwrites 10
    assert buf.pop() == 20
    assert buf.pop() is None


def test_interleaved_push_pop():
    buf = make_buf(3)
    buf.push(1)
    buf.push(2)
    assert buf.pop() == 1
    buf.push(3)
    buf.push(4)
    assert buf.pop() == 2
    assert buf.pop() == 3
    assert buf.pop() == 4
