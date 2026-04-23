import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from backend.api.schemas.agents import VerificationResult
from backend.services.verification.dafny_runner import DafnyRunner


@pytest.fixture
def runner():
    return DafnyRunner(binary_path="dafny", timeout=5, solver_path="z3")


@pytest.mark.asyncio
async def test_verify_empty_source(runner):
    """Test that an empty Dafny specification immediately fails verification without running the subprocess."""
    result = await runner.verify("   \n  ")
    assert result.verified is False
    assert result.prover == "dafny"
    assert "No Dafny specification provided" in result.solver_output


@pytest.mark.asyncio
@patch("backend.services.verification.dafny_runner.asyncio.create_subprocess_exec")
@patch("backend.services.verification.dafny_runner.asyncio.wait_for")
async def test_verify_success(mock_wait, mock_exec, runner):
    """Test that a successful Dafny verification (exit code 0) returns verified=True and no errors."""
    # Mock the process
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_exec.return_value = mock_proc

    # Mock the output (stdout, stderr)
    mock_wait.return_value = (b"Dafny program verifier finished with 1 verified, 0 errors\n", b"")

    result = await runner.verify("method Foo() {}")
    
    assert result.verified is True
    assert result.prover == "dafny"
    assert "1 verified, 0 errors" in result.solver_output
    assert len(result.failing_assertions) == 0


@pytest.mark.asyncio
@patch("backend.services.verification.dafny_runner.asyncio.create_subprocess_exec")
@patch("backend.services.verification.dafny_runner.asyncio.wait_for")
async def test_verify_failure(mock_wait, mock_exec, runner):
    """Test that a failed Dafny verification correctly parses the failing assertions from the output."""
    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_exec.return_value = mock_proc

    stdout = b"""
file.dfy(10,5): Error: A postcondition might not hold on this return path.
file.dfy(2,10): Related location: This is the postcondition that might not hold.
file.dfy(15,8): Error: A precondition for this call could not be proved.
Dafny program verifier finished with 0 verified, 2 errors
"""
    mock_wait.return_value = (stdout, b"")

    result = await runner.verify("method Bad() { assert false; }")
    
    assert result.verified is False
    assert len(result.failing_assertions) == 2
    assert "A postcondition might not hold" in result.failing_assertions[0]
    assert "A precondition for this call could not be proved" in result.failing_assertions[1]


@pytest.mark.asyncio
@patch("backend.services.verification.dafny_runner.asyncio.create_subprocess_exec")
async def test_verify_timeout(mock_exec, runner):
    """Test that verification gracefully handles timeouts by killing the process and returning a failure."""
    from unittest.mock import Mock
    mock_proc = AsyncMock()
    mock_proc.kill = Mock()
    mock_exec.return_value = mock_proc
    
    # Force wait_for to raise a TimeoutError
    with patch("backend.services.verification.dafny_runner.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await runner.verify("method Infinite() {}")
        
    assert result.verified is False
    assert "timed out" in result.solver_output
    mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
@patch("backend.services.verification.dafny_runner.asyncio.create_subprocess_exec", side_effect=FileNotFoundError)
async def test_verify_binary_not_found(mock_exec, runner):
    """Test that missing binary paths (like Dafny not being installed) return a clean failure."""
    result = await runner.verify("method NotFound() {}")
    assert result.verified is False
    assert "Dafny binary not found" in result.solver_output


def test_parse_failing_assertions():
    """Test the regular expressions that extract specific Dafny errors (preconditions, invariants, etc.) from raw text."""
    output = """
foo.dfy(10,2): Error: A precondition for this call could not be proved.
foo.dfy(15,2): Error: A postcondition might not hold on this return path.
foo.dfy(18,2): Error: This loop invariant might not be maintained by the loop.
foo.dfy(20,2): Error: decreases expression might not decrease.
foo.dfy(22,2): Error: This assertion might not hold.
    """
    failures = DafnyRunner._parse_failing_assertions(output)
    assert len(failures) == 5
    assert "precondition" in failures[0]
    assert "postcondition" in failures[1]
    assert "invariant" in failures[2]
    assert "decreases" in failures[3]
    assert "assertion" in failures[4]
