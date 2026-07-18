
# nodes.py — All 13 nodes for Autonomous Python Coding Agent

import os
import ast
import subprocess
import re
import hashlib
import importlib.util
import tempfile

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb

from state import State

# ── LLM ──────────────────────────────────
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# ── CHROMADB ─────────────────────────────
chroma_client     = chromadb.Client()
memory_collection = chroma_client.get_or_create_collection("bug_fixes")

# ─────────────────────────────────────────
# NODE 1 — PLANNER
# ─────────────────────────────────────────
def planner(state: State):
    print("\n📋 Planner thinking...")
    response = llm.invoke([
        SystemMessage(content="You are a coding planner. Break tasks into clear steps."),
        HumanMessage(content=f"""
Break this coding task into clear steps:
Task: {state['task']}
Reply with:
1. What the function should do
2. Input and output format
3. Edge cases to handle
4. Test cases to verify
""")
    ])
    print("Plan ready")
    return {"plan": response.content}

# ─────────────────────────────────────────
# NODE 2 — CODER
# ─────────────────────────────────────────
def coder(state: State):
    print("\n💻 Coder writing code...")

    past_fixes = ""
    if state["error"]:
        try:
            results = memory_collection.query(query_texts=[state["error"]], n_results=2)
            if results["documents"][0]:
                past_fixes = "\n".join(results["documents"][0])
                print("🧠 Found past fixes in memory!")
        except Exception:
            pass

    response = llm.invoke([
        SystemMessage(content="""You are an expert Python developer.
Write clean working Python code WITH type hints on every function.
Return ONLY the code — no explanation, no markdown, no backticks."""),
        HumanMessage(content=f"""
Task: {state['task']}
Plan to follow:
{state['plan']}
Previous error (fix this):
{state['error'] if state['error'] else 'No errors yet — write fresh code'}
Reflection notes:
{state.get('reflection_notes', '') or 'None'}
Past fixes from memory:
{past_fixes if past_fixes else 'No past fixes available'}
Rules:
- Type hints on ALL functions
- Docstring on every function
- Keep it simple and readable
- MUST include demo calls inside: if __name__ == '__main__': that print results
Write complete working Python code only:
""")
    ])

    code = response.content
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```", "", code)
    code = code.strip()

    print(f"Code written ({len(code.splitlines())} lines)")
    return {"code": code, "error": "", "fixed_code": "", "reflection_notes": ""}

# ─────────────────────────────────────────
# NODE 3 — AST VALIDATOR
# ─────────────────────────────────────────
def ast_validator(state: State):
    print("\n🌳 AST Validator checking syntax...")
    code = state["fixed_code"] if state["fixed_code"] else state["code"]

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return {"ast_valid": False, "error": f"SyntaxError at line {e.lineno}: {e.msg}"}

    # Hard-fail on hallucinated imports: catching this here is cheaper than
    # letting it slide through to the Tester, where it would just crash with
    # ModuleNotFoundError anyway and burn a Tester retry instead of an AST one.
    hallucinated_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                try:
                    found = importlib.util.find_spec(base) is not None
                except (ImportError, ValueError, ModuleNotFoundError):
                    found = False
                if not found:
                    hallucinated_imports.append(base)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".")[0]
                try:
                    found = importlib.util.find_spec(base) is not None
                except (ImportError, ValueError, ModuleNotFoundError):
                    found = False
                if not found:
                    hallucinated_imports.append(base)

    if hallucinated_imports:
        unknown = list(set(hallucinated_imports))
        print(f"❌ Hallucinated/unavailable imports: {unknown}")
        return {
            "ast_valid": False,
            "error": f"Unknown or unavailable imports: {unknown}. "
                     f"Rewrite using only the Python standard library or already-imported packages."
        }

    # Missing return type hints stay a soft warning, not a hard failure.
    # Escalating this to a blocker would burn one of only 3 AST retries on a
    # style nitpick, and if AST fails 3 times the pipeline ends early with
    # NO code shown to the user at all — far worse than a missing "-> int".
    missing = [n.name for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and not n.returns and n.name != "__init__"]
    if missing:
        print(f"⚠️ Missing return hints: {missing}")

    print("✅ AST passed!")
    return {"ast_valid": True}

# ─────────────────────────────────────────
# NODE 4 — TEST GENERATOR
# ─────────────────────────────────────────
def test_generator(state: State):
    print("\n🧬 Test Generator creating tests...")
    code = state["fixed_code"] if state["fixed_code"] else state["code"]

    response = llm.invoke([
        SystemMessage(content="""You are a Python testing expert.
Return ONLY runnable Python test code — no markdown, no backticks."""),
        HumanMessage(content=f"""
Generate test cases for this code:
TASK: {state['task']}
CODE:
{code}
Rules:
- Copy ALL function definitions inline — do NOT import from files
- Cover: normal cases, edge cases, large input
- Call each test function at the bottom to run them
- Do NOT use unittest or sys — just plain assert statements
- Print "All tests passed!" at the end if successful
Return ONLY runnable Python code:
""")
    ])

    tests = response.content
    tests = re.sub(r"```python", "", tests)
    tests = re.sub(r"```", "", tests)
    tests = tests.strip()

    print(f"Generated {tests.count('def test_')} test functions")
    return {"generated_tests": tests}

# ─────────────────────────────────────────
# NODE 5 — TESTER
# ─────────────────────────────────────────
def tester(state: State):
    print("\n🧪 Tester running code...")
    code = state["fixed_code"] if state["fixed_code"] else state["code"]

    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            if not result.stdout.strip():
                print("❌ No output produced")
                return {
                    "test_result": "",
                    "error": "Code ran but produced no output. Add print statements in if __name__ == '__main__'.",
                    "passed": False
                }
            print("✅ Code passed!")

            test_output = ""
            if state.get("generated_tests"):
                try:
                    test_run = subprocess.run(
                        ["python", "-c", state["generated_tests"]],
                        capture_output=True, text=True, timeout=15
                    )
                    if test_run.returncode == 0:
                        test_output = "✅ All generated tests passed\n" + test_run.stdout
                    else:
                        test_output = f"⚠️ Some tests failed:\n{test_run.stderr[:200]}"
                except Exception as e:
                    test_output = f"Test run error: {e}"

            # Promote whichever version actually passed into "code" so every
            # downstream node (hypothesis/benchmark/security/complexity/reviewer)
            # keeps working on the tested version instead of silently
            # falling back to the pre-fix original once fixed_code is cleared.
            working_code = code
            return {
                "test_result": result.stdout + "\n" + test_output,
                "error": "",
                "passed": True,
                "code": working_code,
                "fixed_code": ""
            }
        else:
            print(f"❌ Failed: {result.stderr[:80]}")
            return {"test_result": "", "error": result.stderr, "passed": False}

    except subprocess.TimeoutExpired:
        return {"test_result": "", "error": "Timed out after 10 seconds", "passed": False}
    except Exception as e:
        return {"test_result": "", "error": str(e), "passed": False}

# ─────────────────────────────────────────
# NODE 6 — HYPOTHESIS TESTER
# ─────────────────────────────────────────
def hypothesis_tester(state: State):
    print("\n🎲 Hypothesis property-based testing...")
    code = state["fixed_code"] if state["fixed_code"] else state["code"]
    hypothesis_result = "Skipped"

    try:
        response = llm.invoke([
            SystemMessage(content="""You are a Hypothesis testing expert.
Return ONLY runnable Python code — no markdown, no backticks."""),
            HumanMessage(content=f"""
Write Hypothesis property tests for this code:
TASK: {state['task']}
CODE:
{code}
Rules:
- Copy function definitions inline
- Use: from hypothesis import given, settings, strategies as st
- DO NOT use unittest or sys anywhere
- Call test functions directly at the bottom
- Keep to 2 simple property tests only
Return ONLY complete runnable Python code:
""")
        ])

        hyp_code = response.content
        hyp_code = re.sub(r"```python", "", hyp_code)
        hyp_code = re.sub(r"```", "", hyp_code)
        hyp_code = hyp_code.strip()

        result = subprocess.run(
            ["python", "-c", hyp_code],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            print("✅ Hypothesis passed!")
            hypothesis_result = "✅ Property-based tests passed with random inputs"
        else:
            err = result.stderr[:200]
            print(f"⚠️ Hypothesis edge case: {err[:80]}")
            hypothesis_result = f"⚠️ Edge case found: {err}"

    except subprocess.TimeoutExpired:
        hypothesis_result = "⚠️ Timed out — possible infinite loop on edge input"
    except Exception as e:
        hypothesis_result = f"⚠️ Error: {str(e)[:100]}"

    return {"hypothesis_result": hypothesis_result}

# ─────────────────────────────────────────
# NODE 7 — PERFORMANCE BENCHMARKER
# ─────────────────────────────────────────
def performance_benchmarker(state: State):
    print("\n⚡ Benchmarking performance...")
    code = state["fixed_code"] if state["fixed_code"] else state["code"]

    # Find the target function via ast in-process — no need to re-parse
    # inside the subprocess, so no string-embedding/quote-stripping needed.
    try:
        tree = ast.parse(code)
        fn_names = [n.name for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    except SyntaxError:
        fn_names = []

    if not fn_names:
        print("⚡ No benchmarkable function found — skipped")
        return {"benchmark_ms": 0.0}

    fn_name = fn_names[0]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp.write("\n\n")
            tmp.write(
                "import timeit as _t\n"
                f"_f = {fn_name}\n"
                "_ran = False\n"
                "for _call in [lambda: _f(100), lambda: _f('hello'), "
                "lambda: _f([1,2,3,4,5]), lambda: _f('racecar'), lambda: _f(10)]:\n"
                "    try:\n"
                "        _ms = _t.timeit(_call, number=1000) * 1000\n"
                "        print('BENCHMARK:' + str(round(_ms, 2)) + 'ms')\n"
                "        _ran = True\n"
                "        break\n"
                "    except Exception:\n"
                "        continue\n"
                "if not _ran:\n"
                "    print('BENCHMARK:skipped')\n"
            )
            tmp_path = tmp.name

        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=20
        )
        output = result.stdout + result.stderr
        match  = re.search(r"BENCHMARK:([\d.]+)ms", output)
        if match:
            ms = float(match.group(1))
            print(f"⚡ {ms:.2f}ms per 1000 runs")
            if ms > 5000:
                return {
                    "benchmark_ms": ms,
                    "error": f"Too slow: {ms:.0f}ms. Optimize algorithm.",
                    "passed": False
                }
            return {"benchmark_ms": ms}
        return {"benchmark_ms": 0.0}
    except Exception as e:
        print(f"⚠️ Benchmark error: {e}")
        return {"benchmark_ms": 0.0}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ─────────────────────────────────────────
# NODE 8 — DEBUGGER
# ─────────────────────────────────────────
def debugger(state: State):
    print(f"\n🔧 Debugger fixing (attempt {state['retries']+1})...")

    # Fix on top of the most recent attempt (fixed_code), not the stale
    # original — otherwise each retry throws away the previous retry's work.
    base_code = state["fixed_code"] if state["fixed_code"] else state["code"]

    response = llm.invoke([
        SystemMessage(content="""You are a Python debugger.
Fix the exact error. Return ONLY fixed code — no markdown, no backticks."""),
        HumanMessage(content=f"""
CODE:
{base_code}
ERROR:
{state['error']}
Return complete fixed Python code only:
""")
    ])

    fixed = response.content
    fixed = re.sub(r"```python", "", fixed)
    fixed = re.sub(r"```", "", fixed)
    fixed = fixed.strip()

    try:
        stable_id = hashlib.md5(state["error"].encode()).hexdigest()[:8]
        memory_collection.add(
            documents=[f"BUG: {state['error']}\nFIX: {fixed}"],
            ids=[f"fix_{state['retries']}_{stable_id}"]
        )
        print("🧠 Stored in memory!")
    except Exception:
        pass

    return {"fixed_code": fixed, "retries": state["retries"] + 1}

# ─────────────────────────────────────────
# NODE 9 — SECURITY AUDITOR
# ─────────────────────────────────────────
def security_auditor(state: State):
    print("\n🔒 Security check...")
    # final_code isn't set until the reviewer node runs, which is *after*
    # this node in the graph — referencing it here was always a no-op.
    code = state["fixed_code"] if state["fixed_code"] else state["code"]

    dangerous = [
        ("eval(",        "Code execution via eval"),
        ("exec(",        "Code execution via exec"),
        ("os.system(",   "Shell injection risk"),
        ("__import__(",  "Dynamic import risk"),
        ("pickle.loads(","Deserialization attack"),
        ("password =",   "Hardcoded credential"),
        ("api_key =",    "Hardcoded API key"),
    ]

    found = [reason for pattern, reason in dangerous if pattern.lower() in code.lower()]

    if found:
        print(f"❌ Security issues: {found}")
        return {
            "is_secure": False,
            "error": f"Security issues: {found}",
            "security_retries": state["security_retries"] + 1
        }

    print("✅ Security passed!")
    return {"is_secure": True}

# ─────────────────────────────────────────
# NODE 10 — COMPLEXITY JUDGE
# ─────────────────────────────────────────
def complexity_judge(state: State):
    print("\n📊 Complexity check...")
    code  = state["fixed_code"] if state["fixed_code"] else state["code"]
    lines = code.split("\n")
    issues = []

    if len(lines) > 60:
        issues.append(f"Too long: {len(lines)} lines")

    max_indent = max(
        (len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0
    )
    if max_indent > 16:
        issues.append("Too deeply nested")

    try:
        response = llm.invoke([
            HumanMessage(f"Rate complexity 1-10:\n{code}\nReply ONLY a number 1-10.")
        ])
        score = int(re.search(r'\d+', response.content.strip()).group())
    except Exception:
        score = 5

    print(f"Complexity: {score}/10")

    if score > 7 or issues:
        print(f"❌ Too complex: {issues}")
        return {
            "is_simple": False,
            "error": f"Too complex (score {score}/10). Simplify.",
            "complexity_retries": state["complexity_retries"] + 1
        }

    print("✅ Complexity passed!")
    return {"is_simple": True}

# ─────────────────────────────────────────
# NODE 11 — SELF REFLECTION
# ─────────────────────────────────────────
def self_reflection(state: State):
    print("\n🪞 Self Reflection...")
    code = state["fixed_code"] if state["fixed_code"] else state["code"]

    response = llm.invoke([
        SystemMessage(content="""You are a senior Python engineer.
Reply in EXACTLY this format:
CONFIDENCE: <1-10>
APPROVED: <YES or NO>
ISSUES: <list or NONE>
NOTES: <one sentence>"""),
        HumanMessage(content=f"Review this code:\nTASK: {state['task']}\nCODE:\n{code}")
    ])

    reflection = response.content.strip()
    lines_map  = {}
    for line in reflection.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            lines_map[key.strip().upper()] = val.strip()

    try:
        confidence = int(re.search(r'\d+', lines_map.get("CONFIDENCE", "7")).group())
    except Exception:
        confidence = 7

    try:
        approved = "YES" in lines_map.get("APPROVED", "YES").upper()
    except Exception:
        approved = True

    issues_text = lines_map.get("ISSUES", "NONE")
    notes       = lines_map.get("NOTES", "Looks good")
    has_issues  = issues_text.upper() not in ("NONE", "") and bool(issues_text.strip())

    if not approved or (has_issues and confidence < 7):
        print(f"❌ Reflection: confidence {confidence}/10")
        return {
            "reflection_ok":    False,
            "reflection_notes": f"Issues: {issues_text}. {notes}",
            "confidence_score": confidence,
            "error": f"Reflection failed ({confidence}/10): {issues_text}",
            "reflection_retries": state.get("reflection_retries", 0) + 1
        }

    print(f"✅ Reflection approved ({confidence}/10)")
    return {
        "reflection_ok":    True,
        "reflection_notes": notes,
        "confidence_score": confidence
    }

# ─────────────────────────────────────────
# NODE 12 — REVIEWER
# ─────────────────────────────────────────
def reviewer(state: State):
    print("\n✨ Reviewer polishing + explaining...")
    code = state["fixed_code"] if state["fixed_code"] else state["code"]

    response = llm.invoke([
        SystemMessage(content="""You are a senior Python developer and teacher.
Do TWO things and return in EXACTLY this format:
FINAL_CODE:
<complete polished code with docstrings and type hints>
EXPLANATION:
<simple explanation covering: what it does, how it works, time complexity, example usage>
"""),
        HumanMessage(content=f"Polish this code and explain it:\n{code}")
    ])

    content    = response.content
    final_code = ""
    explanation= ""

    if "FINAL_CODE:" in content and "EXPLANATION:" in content:
        parts      = content.split("EXPLANATION:")
        code_part  = parts[0].replace("FINAL_CODE:", "").strip()
        code_part  = re.sub(r"```python", "", code_part)
        code_part  = re.sub(r"```", "", code_part)
        final_code  = code_part.strip()
        explanation = parts[1].strip()
    else:
        final_code  = code
        explanation = content.strip()

    if not explanation:
        explanation = "Code completed successfully. See final code above."

    return {
        "final_code":  final_code,
        "explanation": explanation,
        "review":      "Polished and explained"
    }
# ─────────────────────────────────────────
# NODE 13 — EXPLAINER (passthrough)
# ─────────────────────────────────────────
def explainer(state: State):
    explanation = state.get("explanation")
    if not explanation:
        return {"explanation": "Code completed successfully. See final code above."}
    
    # LangGraph requires a state update. 
    # Re-writing the existing explanation satisfies this rule.
    return {"explanation": explanation}
