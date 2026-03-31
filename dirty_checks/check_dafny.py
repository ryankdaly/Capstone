import asyncio
from backend.services.verification.dafny_runner import DafnyRunner

runner = DafnyRunner()

# Test 1: good spec (should verify)
good = '''
method Max(a: int, b: int) returns (r: int)
ensures r >= a && r >= b
{
if a >= b { r := a; } else { r := b; }
}
'''

# Test 2: bad spec (postcondition should fail)
bad = '''
method Max(a: int, b: int) returns (r: int)
ensures r >= a && r >= b
{
r := a;
}
'''

r1 = asyncio.run(runner.verify(good))
print(f'Good spec: verified={r1.verified}  ({r1.execution_time_seconds}s)')
print(f'  output: {r1.solver_output[:200]}')

r2 = asyncio.run(runner.verify(bad))
print(f'Bad spec:  verified={r2.verified}  ({r2.execution_time_seconds}s)')
print(f'  failures: {r2.failing_assertions}')
