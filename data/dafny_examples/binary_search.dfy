/* REQ: BS-001 — Binary search on sorted array, DO-178C Level A */

// PASS: loop invariants and decreases clause fully established
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

// FAIL: lo := mid instead of lo := mid + 1
//       decreases clause `hi - lo + 1` may not decrease — non-termination
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
      lo := mid;  // BUG: should be mid + 1
    } else {
      hi := mid - 1;
    }
  }
}
