/* REQ: ABS-001 — Absolute value, DO-178C Level A */

// PASS: both postconditions satisfied for all inputs
method AbsoluteValue(x: int) returns (result: int)
  ensures result >= 0
  ensures result == x || result == -x
{
  if x >= 0 { result := x; } else { result := -x; }
}

// FAIL: else branch returns x instead of -x
//       postcondition `result >= 0` does not hold for negative input
method AbsoluteValueBug(x: int) returns (result: int)
  ensures result >= 0
  ensures result == x || result == -x
{
  if x >= 0 { result := x; } else { result := x; }  // BUG: should be -x
}
