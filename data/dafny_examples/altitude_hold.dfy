/* REQ: ALT-001 — Altitude hold with safe range enforcement, DO-178C Level A */
const MIN_ALT: int := 100    // minimum safe altitude in feet
const MAX_ALT: int := 40000  // maximum safe altitude in feet

// PASS: all commanded values are clamped into [MIN_ALT, MAX_ALT]
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

method AltitudeHold(current: int, target: int) returns (output: int)
  requires MIN_ALT <= current <= MAX_ALT
  ensures MIN_ALT <= output <= MAX_ALT
{
  output := ClampAltitude(target);
}

// FAIL: guard uses > instead of < for the lower bound check
//       values below MIN_ALT pass through unclamped — postcondition `safe >= MIN_ALT` fails
method ClampAltitudeBug(commanded: int) returns (safe: int)
  ensures MIN_ALT <= safe <= MAX_ALT
{
  if commanded > MIN_ALT {  // BUG: should be < MIN_ALT
    safe := MIN_ALT;
  } else if commanded > MAX_ALT {
    safe := MAX_ALT;
  } else {
    safe := commanded;
  }
}

method AltitudeHoldBug(current: int, target: int) returns (output: int)
  requires MIN_ALT <= current <= MAX_ALT
  ensures MIN_ALT <= output <= MAX_ALT
{
  output := ClampAltitudeBug(target);
}
