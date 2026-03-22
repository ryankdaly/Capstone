/* REQ: CLAMP-001 — Sensor value clamping, DO-178C Level A */

// PASS: all three branch postconditions satisfied
method Clamp(value: int, lo: int, hi: int) returns (result: int)
  requires lo <= hi
  ensures lo <= result <= hi
  ensures value < lo        ==> result == lo
  ensures value > hi        ==> result == hi
  ensures lo <= value <= hi ==> result == value
{
  if value < lo {
    result := lo;
  } else if value > hi {
    result := hi;
  } else {
    result := value;
  }
}

// FAIL: lo and hi assignments are swapped in the branches
//       all three branch postconditions fail
method ClampBug(value: int, lo: int, hi: int) returns (result: int)
  requires lo <= hi
  ensures lo <= result <= hi
  ensures value < lo        ==> result == lo
  ensures value > hi        ==> result == hi
  ensures lo <= value <= hi ==> result == value
{
  if value < lo {
    result := hi;  // BUG: should be lo
  } else if value > hi {
    result := lo;  // BUG: should be hi
  } else {
    result := value;
  }
}
