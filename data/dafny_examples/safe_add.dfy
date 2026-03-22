/* REQ: SA-001 — Safe 32-bit integer addition, DO-178C Level A */
const INT32_MAX: int := 2147483647
const INT32_MIN: int := -2147483648

// PASS: both overflow directions checked, postconditions fully satisfied
method SafeAdd(a: int, b: int) returns (result: int, ok: bool)
  requires INT32_MIN <= a <= INT32_MAX
  requires INT32_MIN <= b <= INT32_MAX
  ensures ok  ==> result == a + b && INT32_MIN <= result <= INT32_MAX
  ensures !ok ==> (a + b > INT32_MAX || a + b < INT32_MIN)
{
  if a + b > INT32_MAX || a + b < INT32_MIN {
    result := 0;
    ok := false;
  } else {
    result := a + b;
    ok := true;
  }
}

// FAIL: only upper overflow bound checked — lower bound omitted
//       `!ok` postcondition fails for large negative sums
method SafeAddBug(a: int, b: int) returns (result: int, ok: bool)
  requires INT32_MIN <= a <= INT32_MAX
  requires INT32_MIN <= b <= INT32_MAX
  ensures ok  ==> result == a + b && INT32_MIN <= result <= INT32_MAX
  ensures !ok ==> (a + b > INT32_MAX || a + b < INT32_MIN)
{
  if a + b > INT32_MAX {  // BUG: missing || a + b < INT32_MIN
    result := 0;
    ok := false;
  } else {
    result := a + b;
    ok := true;
  }
}
