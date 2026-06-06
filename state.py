# state.py — Shared State for Autonomous Python Coding Agent

from typing import TypedDict


class State(TypedDict):
    task:               str
    plan:               str
    code:               str
    test_result:        str
    error:              str
    fixed_code:         str
    explanation:        str
    review:             str
    final_code:         str
    retries:            int
    security_retries:   int
    complexity_retries: int
    passed:             bool
    is_secure:          bool
    is_simple:          bool
    ast_valid:          bool
    generated_tests:    str
    hypothesis_result:  str
    benchmark_ms:       float
    reflection_ok:      bool
    reflection_notes:   str
    confidence_score:   int
