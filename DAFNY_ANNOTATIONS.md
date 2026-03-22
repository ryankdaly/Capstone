# Dafny Annotations for the Actor Agent

In the HPEMA pipeline, the **Actor** agent is responsible for generating both the safety-critical source code (e.g., C or SPARK Ada) AND its formal specification using [Dafny](https://dafny.org/). 

This document outlines how the Actor models these annotations and what is expected during the generation phase.

## 1. Output Structure

The Actor agent is required to output a JSON object defined by the `CodeCandidate` schema (see `backend/api/schemas/agents.py`). The Dafny specification is returned as a plain-text string in the `dafny_spec` field.

```json
{
  "source_code": "int binary_search(int* arr, int len, int target) { ... }",
  "dafny_spec": "method BinarySearch(arr: array<int>, target: int) returns (index: int)\n  requires arr != null\n  ... ",
  "reasoning_trace": "...",
  "language": "C",
  "annotations": ["no-dynamic-alloc", "bounds-checked"]
}
```

## 2. Required Annotation Types

When prompting the Actor or evaluating its output, the `dafny_spec` must include the following structural elements to mathematically prove the code's safety:

### Preconditions (`requires`)
Defines the conditions that must be true *before* a function is called.
- Examples: Arrays must not be null, lengths must be non-negative, indices must be within bounds.
```dafny
requires arr != null
requires arr.Length > 0
```

### Postconditions (`ensures`)
Defines the guarantees the function provides *after* it executes.
- Examples: The return value correctly reflects the logic, memory was not leaked, output arrays were sorted.
```dafny
ensures 0 <= index < arr.Length ==> arr[index] == target
ensures index == -1 ==> target !in arr[..]
```

### Frame Conditions (`reads` / `modifies`)
Specifies exactly what memory the function is allowed to read from or write to. Essential for proving no unintended side effects.
```dafny
reads arr
modifies arr
```

### Loop Invariants and Termination (`invariant` / `decreases`)
For any loops (e.g., `while` or `for`), the Actor must generate invariants to prove the loop works correctly and terminates.
- `invariant`: A condition that is true before and after every iteration.
- `decreases`: A strictly decreasing metric (like distance between `low` and `high` pointers) to prove the loop will not run infinitely.
```dafny
invariant 0 <= low <= high <= arr.Length
invariant target !in arr[..low] && target !in arr[high..]
decreases high - low
```

## 3. Example: Binary Search

If the user requirement is: *"Implement a binary search function with bounds checking"*, the Actor should generate a `dafny_spec` resembling:

```dafny
method BinarySearch(arr: array<int>, target: int) returns (index: int)
    requires arr != null
    // Array must be sorted
    requires forall i, j :: 0 <= i < j < arr.Length ==> arr[i] <= arr[j]
    
    // Bounds on return value
    ensures -1 <= index < arr.Length
    // Correctness on found
    ensures index >= 0 ==> arr[index] == target
    // Correctness on not found
    ensures index == -1 ==> forall i :: 0 <= i < arr.Length ==> arr[i] != target
{
    var low := 0;
    var high := arr.Length;
    
    while low < high
        invariant 0 <= low <= high <= arr.Length
        invariant forall i :: 0 <= i < low ==> arr[i] != target
        invariant forall i :: high <= i < arr.Length ==> arr[i] != target
        decreases high - low
    {
        var mid := low + (high - low) / 2;
        if arr[mid] < target {
            low := mid + 1;
        } else if arr[mid] > target {
            high := mid;
        } else {
            return mid;
        }
    }
    return -1;
}
```

## 4. How to Update the Actor

If you find the Actor is failing verification due to missing or incorrect Dafny syntax:
1. **Update the Prompt:** Edit `backend/services/llm/prompts/actor.txt` to include explicit examples or stricter rules on `invariant` generation.
2. **Standard Mappings:** If the requirement references a standard (e.g., DO-178C), the Actor will map the Dafny annotations to safety guardrails required by that standard.

## 5. Future Enhancements
Currently, the pipeline translates C/SPARK to Dafny to verify using `dafny_runner.py`. Future iterations may investigate auto-translating Dafny back into the target source language immediately.
