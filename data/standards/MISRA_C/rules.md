# MISRA C:2012 Rules (Public Subset for Safety-Critical C Code)

## Rule 1.3: No Undefined Behavior
There shall be no occurrence of undefined or critical unspecified behavior.

## Rule 8.13: Const Correctness
A pointer should point to a const-qualified type whenever possible.

## Rule 10.1: Operand Types
Operands shall not be of an inappropriate essential type for the operator.

## Rule 11.3: No Cast Between Pointer and Integer
A cast shall not be performed between a pointer to object type and an integral type.

## Rule 12.1: Operator Precedence
The precedence of operators within expressions should be made explicit with parentheses.

## Rule 14.3: No Dead Code
A controlling expression shall not be an invariant expression (no dead code).

## Rule 15.7: Else Required
All if...else if constructs shall be terminated with an else statement.

## Rule 17.2: No Recursion
Functions shall not call themselves, either directly or indirectly.

## Rule 17.7: Return Value Used
The value returned by a function having non-void return type shall be used.

## Rule 21.3: No Dynamic Memory
The memory allocation and deallocation functions of <stdlib.h> shall not be used (malloc, calloc, realloc, free).

## Rule 21.6: No Standard I/O
The Standard Library input/output functions shall not be used in production safety-critical code.
