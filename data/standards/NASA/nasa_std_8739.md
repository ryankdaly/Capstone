# NASA-STD-8739.8 — Software Assurance and Software Safety Standard

## 4.1 Software Classification
Software shall be classified based on its criticality:
- Class A: Software whose failure may cause loss of life or vehicle
- Class B: Software whose failure may cause major mission impact
- Class C: Software whose failure causes minor mission impact

## 4.2 Software Safety Requirements
For Class A software:
a) Hazard analysis shall identify software contributions to system hazards.
b) Safety-critical requirements shall be formally specified.
c) Independent verification of safety-critical code is required.

## 4.3 Coding Standards
a) All code shall be developed using approved coding standards.
b) Code complexity metrics (cyclomatic complexity) shall not exceed defined thresholds.
c) Functions shall have a single entry and single exit point.
d) Global variables shall be minimized and their usage documented.

## 4.4 Testing Requirements
a) Requirements-based testing: every requirement shall have at least one test case.
b) Boundary value testing for all input parameters.
c) Robustness testing with out-of-range and invalid inputs.
d) Coverage analysis: statement coverage (minimum), branch coverage (Class A/B).

## 4.5 Formal Methods
For Class A software, the use of formal methods is strongly recommended:
a) Formal specification of critical algorithms
b) Model checking or theorem proving for critical properties
c) Static analysis with formally-defined checkers
