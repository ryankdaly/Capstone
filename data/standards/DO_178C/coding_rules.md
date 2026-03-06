# DO-178C Coding Rules (Derived from Level A Requirements)

## Rule DO-178C-CR-001: No Dynamic Memory Allocation
Safety-critical software shall not use dynamic memory allocation (malloc, calloc, realloc, free) after initialization. All memory shall be statically allocated.
Rationale: Dynamic allocation introduces non-deterministic behavior and fragmentation risks.

## Rule DO-178C-CR-002: No Recursion
Recursive function calls are prohibited. All loops must have provable upper bounds.
Rationale: Recursion risks stack overflow, which is undetectable at compile time.

## Rule DO-178C-CR-003: Bounded Loop Iterations
All loops shall have a statically determinable upper bound on iterations. This bound shall be documented and verifiable.
Rationale: Unbounded loops can cause watchdog timeouts or missed deadlines in real-time systems.

## Rule DO-178C-CR-004: Defensive Coding
All function inputs shall be validated against their specified ranges before use. Functions shall handle out-of-range inputs gracefully (e.g., clamp, return error code).
Rationale: Invalid inputs from sensors or other components must not cause undefined behavior.

## Rule DO-178C-CR-005: No Undefined Behavior
Code shall not rely on behavior that is undefined by the language standard. This includes:
- Signed integer overflow
- Null pointer dereference
- Use of uninitialized variables
- Out-of-bounds array access

## Rule DO-178C-CR-006: Traceability Annotations
Every function shall include a comment or annotation tracing it to its originating requirement.
Format: /* REQ: <requirement-id> */

## Rule DO-178C-CR-007: Formal Specification Required (Level A)
For Design Assurance Level A software, formal specifications (preconditions, postconditions, invariants) shall be provided for all safety-critical functions.

## Rule DO-178C-CR-008: Independent Verification
Code shall be structured to facilitate independent review. This means:
- Clear separation of concerns
- Minimal side effects
- Explicit data flow (no hidden global state)
