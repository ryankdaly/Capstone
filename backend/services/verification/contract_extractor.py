"""Extract requires/ensures clauses from a verified Dafny method.

The output feeds into the Checker agent, which translates Dafny postconditions
into executable Python assertions — closing the formal verification loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DafnyContracts:
    """Formal contracts extracted from a verified Dafny method."""

    method_name: str = ""
    params: str = ""
    requires: list[str] = field(default_factory=list)
    ensures: list[str] = field(default_factory=list)

    @property
    def has_contracts(self) -> bool:
        return bool(self.requires or self.ensures)


def extract_contracts(dafny_source: str) -> DafnyContracts:
    """Parse requires/ensures clauses from the first Dafny method in the source.

    Handles multi-line clauses joined by indented continuation lines.
    Returns an empty DafnyContracts if no method is found.
    """
    contracts = DafnyContracts()

    # Find first method signature
    sig_match = re.search(
        r"\bmethod\s+(\w+)\s*\(([^)]*)\)",
        dafny_source,
        re.DOTALL,
    )
    if not sig_match:
        return contracts

    contracts.method_name = sig_match.group(1)
    contracts.params = re.sub(r"\s+", " ", sig_match.group(2)).strip()

    # Scan from end of signature to opening brace for the body
    spec_start = sig_match.end()
    body_open = dafny_source.find("{", spec_start)
    spec_region = (
        dafny_source[spec_start:body_open]
        if body_open != -1
        else dafny_source[spec_start:]
    )

    _parse_clauses(spec_region, contracts)
    return contracts


_CLAUSE_RE = re.compile(r"^\s*(requires|ensures)\s+(.*)", re.IGNORECASE)
# A continuation line: indented ≥4 spaces and not starting a new keyword
_CONTINUATION_RE = re.compile(r"^\s{4,}\S")


def _parse_clauses(region: str, contracts: DafnyContracts) -> None:
    """Populate contracts.requires / contracts.ensures from the spec region."""
    current_key: str | None = None
    current_clause: list[str] = []

    def _flush() -> None:
        if current_key and current_clause:
            clause = " ".join(current_clause).strip()
            if clause:
                getattr(contracts, current_key).append(clause)

    for line in region.splitlines():
        m = _CLAUSE_RE.match(line)
        if m:
            _flush()
            current_key = m.group(1).lower()
            current_clause = [m.group(2).strip()]
        elif current_key and _CONTINUATION_RE.match(line):
            current_clause.append(line.strip())
        else:
            _flush()
            current_key = None
            current_clause = []

    _flush()
