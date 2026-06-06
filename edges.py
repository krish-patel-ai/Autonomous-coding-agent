# edges.py — Conditional edge routing for Autonomous Python Coding Agent

from langgraph.graph import END
from state import State


def route_after_ast(state: State) -> str:
    if state["ast_valid"]:
        return "test_generator"
    if state["retries"] >= 3:
        return "__end__"
    return "debugger"


def route_after_test(state: State) -> str:
    if state["passed"]:
        return "hypothesis"
    if state["retries"] >= 3:
        return "hypothesis"  # move forward after 3 tries
    return "debugger"


def route_after_hypothesis(state: State) -> str:
    return "benchmark"  # never blocks pipeline


def route_after_benchmark(state: State) -> str:
    error = state.get("error", "")
    if "too slow" in error.lower() or "optimize" in error.lower():
        if state["retries"] >= 3:
            return "security"
        return "debugger"
    return "security"


def route_after_security(state: State) -> str:
    if state["is_secure"]:
        return "complexity"
    if state["security_retries"] >= 2:
        return "complexity"  # give up, move forward
    return "coder"


def route_after_complexity(state: State) -> str:
    if state["is_simple"]:
        return "reflection"
    if state["complexity_retries"] >= 2:
        return "reflection"  # give up, move forward
    return "coder"


def route_after_reflection(state: State) -> str:
    if state["reflection_ok"]:
        return "reviewer"
    if state["retries"] >= 3:
        return "reviewer"  # ship after 3 tries
    return "coder"
