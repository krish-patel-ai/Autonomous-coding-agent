# main.py — Graph builder for Autonomous Python Coding Agent

import os
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langgraph.graph import StateGraph, END

from state import State
from nodes import (
    planner, coder, ast_validator, test_generator,
    tester, hypothesis_tester, performance_benchmarker,
    debugger, security_auditor, complexity_judge,
    self_reflection, reviewer, explainer
)
from edges import (
    route_after_ast, route_after_test, route_after_hypothesis,
    route_after_benchmark, route_after_security,
    route_after_complexity, route_after_reflection
)

# ── BUILD GRAPH ───────────────────────────
def build_graph():
    builder = StateGraph(State)

    # Add all 13 nodes
    builder.add_node("planner",        planner)
    builder.add_node("coder",          coder)
    builder.add_node("ast_validator",  ast_validator)
    builder.add_node("test_generator", test_generator)
    builder.add_node("tester",         tester)
    builder.add_node("hypothesis",     hypothesis_tester)
    builder.add_node("benchmark",      performance_benchmarker)
    builder.add_node("debugger",       debugger)
    builder.add_node("security",       security_auditor)
    builder.add_node("complexity",     complexity_judge)
    builder.add_node("reflection",     self_reflection)
    builder.add_node("reviewer",       reviewer)
    builder.add_node("explainer",      explainer)

    # Entry point
    builder.set_entry_point("planner")

    # Fixed edges
    builder.add_edge("planner",        "coder")
    builder.add_edge("coder",          "ast_validator")
    builder.add_edge("test_generator", "tester")
    builder.add_edge("debugger",       "ast_validator")
    builder.add_edge("reviewer",       "explainer")
    builder.add_edge("explainer",      END)

    # Conditional edges
    builder.add_conditional_edges("ast_validator", route_after_ast,
        {"test_generator": "test_generator", "debugger": "debugger", "__end__": END})
    builder.add_conditional_edges("tester", route_after_test,
        {"hypothesis": "hypothesis", "debugger": "debugger"})
    builder.add_conditional_edges("hypothesis", route_after_hypothesis,
        {"benchmark": "benchmark"})
    builder.add_conditional_edges("benchmark", route_after_benchmark,
        {"security": "security", "debugger": "debugger"})
    builder.add_conditional_edges("security", route_after_security,
        {"complexity": "complexity", "coder": "coder"})
    builder.add_conditional_edges("complexity", route_after_complexity,
        {"reflection": "reflection", "coder": "coder"})
    builder.add_conditional_edges("reflection", route_after_reflection,
        {"reviewer": "reviewer", "coder": "coder"})

    return builder.compile()

# ── COMPILED GRAPH ────────────────────────
graph = build_graph()

# ── INITIAL STATE ─────────────────────────
def get_initial_state(task: str) -> dict:
    return {
        "task":               task,
        "plan":               "",
        "code":               "",
        "test_result":        "",
        "error":              "",
        "fixed_code":         "",
        "explanation":        "",
        "review":             "",
        "final_code":         "",
        "retries":            0,
        "security_retries":   0,
        "complexity_retries": 0,
        "passed":             False,
        "is_secure":          False,
        "is_simple":          False,
        "ast_valid":          False,
        "generated_tests":    "",
        "hypothesis_result":  "",
        "benchmark_ms":       0.0,
        "reflection_ok":      False,
        "reflection_notes":   "",
        "confidence_score":   0,
    }

# ── RUN ───────────────────────────────────
if __name__ == "__main__":
    tasks = [
        "Write a Python function to find all prime numbers up to n",
        "Write a Python function to check if a string is a palindrome",
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"📋 Task: {task}")
        print("="*60)

        result = graph.invoke(get_initial_state(task), {"recursion_limit": 50})

        print(f"\n{'='*60}")
        print(f"💻 Final Code:\n{result['final_code'] or result['code']}")
        print(f"\n📖 Explanation:\n{result['explanation']}")
        bms  = result['benchmark_ms']
        conf = result['confidence_score']
        print(f"\n🧪 Tests:      {result['test_result'][:100]}")
        print(f"🎲 Hypothesis: {result['hypothesis_result']}")
        print(f"⚡ Speed:      {bms:.1f}ms" if bms > 0 else "⚡ Speed:      Skipped")
        print(f"🪞 Confidence: {conf}/10" if conf > 0 else "🪞 Confidence: 7/10 (default)")
        print(f"🔒 Secure:     {result['is_secure']}")
        print(f"📊 Simple:     {result['is_simple']}")
        print(f"🔄 Retries:    {result['retries']}")
