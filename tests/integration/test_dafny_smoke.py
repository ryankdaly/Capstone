"""Smoke tests for DafnyRunner against a real Dafny binary.

These tests require `dafny` to be on PATH and are automatically skipped
if it is not installed. They complement the mocked unit tests in
tests/unit/test_dafny_runner.py by exercising the full subprocess path.
"""

from __future__ import annotations

import shutil

import pytest

from backend.services.verification.dafny_runner import DafnyRunner

# Applied to every test that needs the real binary.
dafny_installed = pytest.mark.skipif(
    shutil.which("dafny") is None,
    reason="dafny binary not on PATH — install Dafny to run smoke tests",
)

# ---------------------------------------------------------------------------
# Fixture source strings (single method only — taken from tests/fixtures/dafny/)
# Each file contains PASS + FAIL methods together, so we pass them individually
# to get clean verified=True / verified=False results.
# ---------------------------------------------------------------------------

ABSOLUTE_VALUE_PASS = """\
method AbsoluteValue(x: int) returns (result: int)
  ensures result >= 0
  ensures result == x || result == -x
{
  if x >= 0 { result := x; } else { result := -x; }
}
"""

ABSOLUTE_VALUE_FAIL = """\
method AbsoluteValueBug(x: int) returns (result: int)
  ensures result >= 0
  ensures result == x || result == -x
{
  if x >= 0 { result := x; } else { result := x; }
}
"""

BINARY_SEARCH_PASS = """\
method BinarySearch(a: array<int>, target: int) returns (index: int)
  requires a.Length >= 0
  requires forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]
  ensures index == -1 || (0 <= index < a.Length && a[index] == target)
{
  var lo := 0;
  var hi := a.Length - 1;
  index := -1;
  while lo <= hi
    invariant 0 <= lo <= a.Length
    invariant -1 <= hi <= a.Length - 1
    invariant forall k :: 0 <= k < lo ==> a[k] != target
    invariant forall k :: hi < k < a.Length ==> a[k] != target
    decreases hi - lo + 1
  {
    var mid := lo + (hi - lo) / 2;
    if a[mid] == target {
      index := mid;
      return;
    } else if a[mid] < target {
      lo := mid + 1;
    } else {
      hi := mid - 1;
    }
  }
}
"""

ALTITUDE_CONTROLLER_PASS = """\
const MIN_ALT: int := 100
const MAX_ALT: int := 40000

method ClampAltitude(commanded: int) returns (safe: int)
  ensures MIN_ALT <= safe <= MAX_ALT
{
  if commanded < MIN_ALT {
    safe := MIN_ALT;
  } else if commanded > MAX_ALT {
    safe := MAX_ALT;
  } else {
    safe := commanded;
  }
}

method AltitudeController(current: int, target: int) returns (output: int)
  requires MIN_ALT <= current <= MAX_ALT
  ensures MIN_ALT <= output <= MAX_ALT
{
  output := ClampAltitude(target);
}
"""

ALTITUDE_CONTROLLER_FAIL = """\
const MIN_ALT: int := 100
const MAX_ALT: int := 40000

method ClampAltitudeBug(commanded: int) returns (safe: int)
  ensures MIN_ALT <= safe <= MAX_ALT
{
  if commanded > MIN_ALT {
    safe := MIN_ALT;
  } else if commanded > MAX_ALT {
    safe := MAX_ALT;
  } else {
    safe := commanded;
  }
}

method AltitudeControllerBug(current: int, target: int) returns (output: int)
  requires MIN_ALT <= current <= MAX_ALT
  ensures MIN_ALT <= output <= MAX_ALT
{
  output := ClampAltitudeBug(target);
}
"""

ARRAY_BOUNDS_PASS = """\
method SafeRead(a: array<int>, index: int, default: int) returns (value: int)
  requires a.Length >= 0
  ensures 0 <= index < a.Length ==> value == a[index]
  ensures (index < 0 || index >= a.Length) ==> value == default
{
  if index < 0 || index >= a.Length {
    value := default;
  } else {
    value := a[index];
  }
}
"""

ARRAY_BOUNDS_FAIL = """\
method SafeReadBug(a: array<int>, index: int, default: int) returns (value: int)
  requires a.Length >= 0
  ensures 0 <= index < a.Length ==> value == a[index]
  ensures (index < 0 || index >= a.Length) ==> value == default
{
  if index < 0 || index >= a.Length {
    value := default;
  } else {
    value := a[index + 1];
  }
}
"""

BINARY_SEARCH_FAIL = """\
method BinarySearchBug(a: array<int>, target: int) returns (index: int)
  requires a.Length >= 0
  requires forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]
  ensures index == -1 || (0 <= index < a.Length && a[index] == target)
{
  var lo := 0;
  var hi := a.Length - 1;
  index := -1;
  while lo <= hi
    invariant 0 <= lo <= a.Length
    invariant -1 <= hi <= a.Length - 1
    invariant forall k :: 0 <= k < lo ==> a[k] != target
    invariant forall k :: hi < k < a.Length ==> a[k] != target
    decreases hi - lo + 1
  {
    var mid := lo + (hi - lo) / 2;
    if a[mid] == target {
      index := mid;
      return;
    } else if a[mid] < target {
      lo := mid;
    } else {
      hi := mid - 1;
    }
  }
}
"""

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner():
    return DafnyRunner(binary_path="dafny", timeout=60)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_source(runner):
    """Empty source short-circuits before spawning a subprocess."""
    result = await runner.verify("")
    assert result.verified is False
    assert "No Dafny specification provided" in result.solver_output


@dafny_installed
@pytest.mark.asyncio
async def test_pass_absolute_value(runner):
    """Correct AbsoluteValue spec verifies cleanly."""
    result = await runner.verify(ABSOLUTE_VALUE_PASS)
    assert result.verified is True, f"Expected verified=True.\nDafny output:\n{result.solver_output}"


@dafny_installed
@pytest.mark.asyncio
async def test_fail_absolute_value(runner):
    """Buggy AbsoluteValue (returns x instead of -x) fails verification."""
    result = await runner.verify(ABSOLUTE_VALUE_FAIL)
    assert result.verified is False, f"Expected verified=False.\nDafny output:\n{result.solver_output}"


@dafny_installed
@pytest.mark.asyncio
async def test_pass_binary_search(runner):
    """Correct BinarySearch spec with loop invariants and decreases verifies cleanly."""
    result = await runner.verify(BINARY_SEARCH_PASS)
    assert result.verified is True, f"Expected verified=True.\nDafny output:\n{result.solver_output}"


@dafny_installed
@pytest.mark.asyncio
async def test_fail_binary_search(runner):
    """Buggy BinarySearch (lo := mid instead of mid+1) fails — non-termination caught by decreases."""
    result = await runner.verify(BINARY_SEARCH_FAIL)
    assert result.verified is False, f"Expected verified=False.\nDafny output:\n{result.solver_output}"


@dafny_installed
@pytest.mark.asyncio
async def test_pass_altitude_controller(runner):
    """Correct AltitudeController with ClampAltitude verifies cleanly."""
    result = await runner.verify(ALTITUDE_CONTROLLER_PASS)
    assert result.verified is True, f"Expected verified=True.\nDafny output:\n{result.solver_output}"


@dafny_installed
@pytest.mark.asyncio
async def test_fail_altitude_controller(runner):
    """Buggy ClampAltitudeBug (> instead of < on lower bound) fails — values below MIN_ALT pass through unclamped."""
    result = await runner.verify(ALTITUDE_CONTROLLER_FAIL)
    assert result.verified is False, f"Expected verified=False.\nDafny output:\n{result.solver_output}"


@dafny_installed
@pytest.mark.asyncio
async def test_pass_array_bounds(runner):
    """Correct SafeRead with bounds check verifies cleanly."""
    result = await runner.verify(ARRAY_BOUNDS_PASS)
    assert result.verified is True, f"Expected verified=True.\nDafny output:\n{result.solver_output}"


@dafny_installed
@pytest.mark.asyncio
async def test_fail_array_bounds(runner):
    """Buggy SafeReadBug (off-by-one: a[index+1]) fails — postcondition value == a[index] violated."""
    result = await runner.verify(ARRAY_BOUNDS_FAIL)
    assert result.verified is False, f"Expected verified=False.\nDafny output:\n{result.solver_output}"
