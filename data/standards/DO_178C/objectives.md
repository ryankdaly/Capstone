# DO-178C Software Considerations in Airborne Systems and Equipment Certification

## Section 6.3 — Software Coding and Integration Process

### 6.3.1 Source Code Development
The source code shall implement the software architecture and low-level requirements. The following objectives apply:

a) Source code shall be traceable to low-level requirements.
b) Source code shall conform to defined coding standards.
c) Source code shall be accurate and consistent with the low-level requirements.
d) Source code shall be verifiable — written in a manner that facilitates verification activities.

### 6.3.2 Coding Standards
Coding standards shall address:
- Use of coding constructs that are consistent with safety objectives
- Restrictions on the use of constructs that may produce unintended behavior (e.g., dynamic memory allocation, recursion, pointer arithmetic)
- Naming conventions and code formatting for readability
- Restrictions on compiler-specific or hardware-specific extensions

### 6.3.3 Code Reviews
Code reviews shall verify:
- Compliance with coding standards
- Traceability to low-level requirements
- Accuracy of code logic
- Correctness of parameter passing and data coupling

## Section 6.4 — Formal Methods (Supplement DO-333)

### 6.4.1 Use of Formal Methods
When formal methods are used as a verification technique:
a) The formal specification shall be traceable to the software requirements.
b) The formal verification shall demonstrate that the source code satisfies the formal specification.
c) Tool qualification may be required for formal verification tools (DO-330).

### 6.4.2 Formal Proofs
A formal proof demonstrates that a property holds for ALL possible executions of the software, not just tested scenarios. This provides a higher level of assurance than testing alone.

## Section 6.7 — Software Verification Process

### 6.7.1 Verification Objectives
- Requirements-based testing shall demonstrate that the software satisfies its requirements.
- Structural coverage analysis shall confirm that the code structure has been exercised.
- For Level A software, Modified Condition/Decision Coverage (MC/DC) is required.

### 6.7.2 Independence of Verification
For Level A and B software, verification activities shall have independence from development activities (the person who writes the code should not be the sole verifier).
