# DO-178C Software Testing Requirements

## Section 6.4 — Software Testing Process

### 6.4.1 Testing Objectives
Software testing shall be performed to demonstrate with high confidence that:
a) The software correctly implements the software requirements.
b) The software does not exhibit unintended behavior.
c) No erroneous or unexpected inputs cause uncontrolled system behavior.

### 6.4.2 Requirements-Based Testing
Test cases shall be derived from software requirements. Each requirement shall have at least one corresponding test case. Requirements-based testing covers:
- Normal range inputs
- Boundary values and equivalence classes
- Robustness: out-of-range, invalid, and null inputs

### 6.4.3 Structural Coverage Analysis
Structural coverage ensures the test cases exercise all code paths:

**Level A (Design Assurance Level A):**
- Modified Condition/Decision Coverage (MC/DC): Every condition in a decision independently affects the outcome. Each entry and exit point exercised at least once.

**Level B:**
- Decision Coverage: Every branch (true/false) exercised at least once.

**Level C:**
- Statement Coverage: Every executable statement exercised at least once.

### 6.4.4 Test Independence
For Level A and B software, testing activities shall be independent from development activities. The same person who wrote the code shall not be the sole author of its tests.

## Section 6.5 — Test Case Design Rules

### 6.5.1 Boundary Value Testing
For any function accepting inputs in range [min, max]:
- Test at min, min+1, max-1, max (boundary values)
- Test just below min and just above max (out-of-range robustness)
- Test a typical middle value (nominal case)

### 6.5.2 Equivalence Partitioning
Inputs shall be grouped into equivalence classes where all values in a class are expected to be treated the same. At minimum one test case per partition.

### 6.5.3 Error Handling Verification
Tests shall verify that:
- Invalid inputs produce defined error responses (not undefined behavior)
- Error return codes are correctly set and returned to the caller
- No silent data corruption occurs on invalid input

## Section 6.6 — Forbidden Constructs Under Testing

The following constructs shall be flagged during test review:
- Functions that cannot be tested in isolation (excessive coupling)
- Test cases that always pass regardless of implementation (trivially true assertions)
- Tests that depend on execution order or shared mutable state
- Tests that use hardcoded values without corresponding requirements traceability

## Appendix A — Test Traceability Matrix
Every test case shall trace to:
1. A specific software requirement identifier
2. The code element under test (function name, line range)
3. The expected result and pass/fail criteria
