# Boeing Software Design Practices (Synthetic — Based on Public Standards)

## SDP-001: Variable Naming
All variables shall use descriptive names. Single-character variable names are prohibited except for loop counters (i, j, k).

## SDP-002: Function Length
No function shall exceed 75 lines of executable code. Functions exceeding this limit shall be decomposed.

## SDP-003: Cyclomatic Complexity
Cyclomatic complexity of any function shall not exceed 10. Functions above this threshold require architectural review.

## SDP-004: Error Handling
All functions that can fail shall return an error code. Error codes shall be checked at every call site. Unhandled errors are prohibited.

## SDP-005: Range Checking
All sensor inputs shall be range-checked against physical limits before use. Out-of-range values shall trigger a fault response (default to safe state).

## SDP-006: Units Documentation
All physical quantities shall have their units documented in comments or type annotations. Mixed-unit arithmetic without explicit conversion is prohibited.

## SDP-007: Watchdog Compliance
All control loops shall complete within their allocated time slice. Worst-case execution time (WCET) shall be analyzed and documented.

## SDP-008: Configuration Management
All code changes shall be traceable to an approved change request. Unauthorized modifications are prohibited.

## SDP-009: Peer Review
All safety-critical code shall undergo peer review by at least one qualified reviewer before integration. Review findings shall be documented.

## SDP-010: Defensive Programming
All array accesses shall be bounds-checked. All pointer dereferences shall be null-checked. All arithmetic operations on safety-critical values shall be overflow-checked.
