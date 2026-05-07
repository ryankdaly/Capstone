"""Edge tests: state_machine — L3."""

import pytest
from conftest import load_artifact

m = load_artifact()

STATES = {"IDLE", "ARMED", "ACTIVE", "SAFE"}
TRANSITIONS = {
    "IDLE": {"ARMED"},
    "ARMED": {"ACTIVE", "IDLE"},
    "ACTIVE": {"SAFE", "IDLE"},
    "SAFE": {"IDLE"},
}


def make_sm(initial="IDLE"):
    return m.SafeStateMachine(STATES, TRANSITIONS, initial)


def test_initial_state():
    sm = make_sm("IDLE")
    assert sm.current_state == "IDLE"


def test_valid_transition():
    sm = make_sm("IDLE")
    sm.transition("ARMED")
    assert sm.current_state == "ARMED"


def test_valid_transition_chain():
    sm = make_sm("IDLE")
    sm.transition("ARMED")
    sm.transition("ACTIVE")
    sm.transition("SAFE")
    assert sm.current_state == "SAFE"


def test_illegal_direct_jump_raises():
    sm = make_sm("IDLE")
    with pytest.raises(Exception):  # InvalidTransition or ValueError
        sm.transition("ACTIVE")  # not in IDLE's allowed set


def test_transition_to_unknown_state_raises():
    sm = make_sm("IDLE")
    with pytest.raises(Exception):
        sm.transition("UNKNOWN_STATE")


def test_invalid_initial_state_raises():
    with pytest.raises((ValueError, KeyError, Exception)):
        m.SafeStateMachine(STATES, TRANSITIONS, "NOT_A_STATE")


def test_current_state_is_read_only():
    sm = make_sm("IDLE")
    with pytest.raises((AttributeError, TypeError)):
        sm.current_state = "ARMED"
