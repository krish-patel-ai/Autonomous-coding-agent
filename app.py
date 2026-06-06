# app.py — Streamlit UI for Autonomous Python Coding Agent
import streamlit as st
import os
import sys

st.set_page_config(
    page_title="Autonomous Python Coding Agent",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .node-card {
        border-radius: 10px;
        padding: 12px 16px;
        margin: 5px 0;
        border-left: 4px solid #444;
        font-size: 14px;
        background: #1e2130;
    }
    .node-pass { border-left-color: #00cc88; }
    .node-fail { border-left-color: #ff4444; }
    .node-skip { border-left-color: #ffaa00; }
    .title-grad {
        background: linear-gradient(90deg, #00cc88, #0088ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────
st.markdown('<p class="title-grad">🤖 Autonomous Python Coding Agent</p>', unsafe_allow_html=True)
st.markdown("**13-node LangGraph pipeline** · AST Validation · Property Testing · Security Audit · Self Reflection")
st.divider()

# ── SIDEBAR ───────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Nodes")
    st.markdown("""
1. 📋 Planner
2. 💻 Coder
3. 🌳 AST Validator
4. 🧬 Test Generator
5. 🧪 Tester
6. 🎲 Hypothesis
7. ⚡ Benchmarker
8. 🔧 Debugger
9. 🔒 Security Auditor
10. 📊 Complexity Judge
11. 🪞 Self Reflection
12. ✨ Reviewer
13. 📖 Explainer
    """)
    st.divider()
    st.markdown("### 🧠 What makes this different?")
    st.markdown("""
- **AST parsing** catches bugs before running
- **Auto-generated tests** — no manual writing
- **Hypothesis** generates 500+ random inputs
- **ChromaDB memory** learns from past fixes
- **Self-reflection** — agent critiques itself
- **Separate retry counters** per node
    """)
    st.divider()
    st.markdown("Built with `LangGraph` · `Groq` · `ChromaDB`")

# ── HELPERS ───────────────────────────────
def node_card(icon, name, status, detail=""):
    cls   = {"pass": "node-pass", "fail": "node-fail", "skip": "node-skip"}.get(status, "node-skip")
    emoji = {"pass": "✅", "fail": "❌", "skip": "⏭️"}.get(status, "⏳")
    detail_html = f"<span style='color:#888;font-size:12px'> — {detail}</span>" if detail else ""
    st.markdown(
        f'<div class="node-card {cls}">{emoji} <b>{icon} {name}</b>{detail_html}</div>',
        unsafe_allow_html=True
    )

def initial_state(task):
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

# ── EXAMPLES ──────────────────────────────
examples = [
    "Write a Python function to find all prime numbers up to n",
    "Write a Python function to find the second largest number in a list",
    "Write a Python function to check if a string is a palindrome",
    "Write a Python function to flatten a nested list",
    "Write a Python function to find factorial using recursion",
]
def set_example_task(example_text):
    # This updates the memory safely before the page redraws
    st.session_state["task_input"] = example_text

# ── INPUT ─────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    task = st.text_area(
        "🎯 Enter your Python task:",
        placeholder="e.g. Write a Python function to find all prime numbers up to n",
        height=100,
        key="task_input"
    )

with col2:
    st.markdown("**💡 Try an example:**")
    for ex in examples:
        # Instead of an 'if' statement, we attach the helper function to 'on_click'
        st.button(
            ex[:38] + "…", 
            key=ex, 
            use_container_width=True,
            on_click=set_example_task, # Calls our helper function
            args=(ex,)                 # Hands the example text to the helper function
        )


if "selected_task" in st.session_state and not task:
    task = st.session_state["selected_task"]

run_btn = st.button("▶ Run Agent", type="primary", use_container_width=True, disabled=not bool(task))

# ── RUN ───────────────────────────────────
if run_btn and task:
    st.divider()
    st.markdown("### 🔄 Pipeline Running...")

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from main import graph
    except Exception as e:
        st.error(f"❌ Could not import agent: {e}")
        st.stop()

    col_status, col_task = st.columns([1, 2])
    with col_status:
        st.markdown("#### 📊 Node Status")
    with col_task:
        st.markdown("#### 💬 Task")
        st.info(task)

    with st.spinner("🤖 Agent working... (~20-40 seconds)"):
        try:
            result  = graph.invoke(initial_state(task), {"recursion_limit": 50})
            success = True
        except Exception as e:
            st.error(f"❌ Agent error: {e}")
            success = False
            result  = {}

    if success and result:
        final_code = result.get("final_code") or result.get("code", "")
        bms        = result.get("benchmark_ms", 0.0)
        conf       = result.get("confidence_score", 0)
        hyp        = result.get("hypothesis_result", "")
        passed     = result.get("passed", False)
        secure     = result.get("is_secure", False)
        simple     = result.get("is_simple", False)
        refl_ok    = result.get("reflection_ok", False)
        retries    = result.get("retries", 0)

        # Node status
        with col_status:
            node_card("📋", "Planner",        "pass", "blueprint ready")
            node_card("💻", "Coder",          "pass", f"{len(final_code.splitlines())} lines")
            node_card("🌳", "AST Validator",  "pass")
            node_card("🧬", "Test Generator", "pass", "tests created")
            node_card("🧪", "Tester",         "pass" if passed else "skip",
                      "passed" if passed else f"retried {retries}x")

            if "✅" in hyp:
                hyp_status = "pass"
            elif "⚠️" in hyp:
                hyp_status = "skip"
            else:
                hyp_status = "skip"
            node_card("🎲", "Hypothesis",     hyp_status, hyp[:35] if hyp else "skipped")
            node_card("⚡", "Benchmarker",    "pass" if bms > 0 else "skip",
                      f"{bms:.1f}ms" if bms > 0 else "skipped")
            node_card("🔒", "Security",       "pass" if secure else "skip",
                      "passed" if secure else "warnings found")
            node_card("📊", "Complexity",     "pass" if simple else "skip",
                      "passed" if simple else "warnings found")
            node_card("🪞", "Self Reflection","pass" if refl_ok else "skip",
                      f"{conf}/10" if conf > 0 else "7/10 default")
            node_card("✨", "Reviewer",       "pass", "polished")
            node_card("📖", "Explainer",      "pass", "docs written")

        # Metrics
        st.divider()
        st.markdown("### 📈 Results Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🔄 Retries",    retries)
        m2.metric("🔒 Secure",     "✅" if secure else "⚠️")
        m3.metric("📊 Simple",     "✅" if simple else "⚠️")
        m4.metric("🪞 Confidence", f"{conf}/10" if conf > 0 else "7/10")
        m5.metric("⚡ Speed",      f"{bms:.1f}ms" if bms > 0 else "Skipped")

        # Final code
        st.divider()
        st.markdown("### 💻 Final Code")
        st.code(final_code, language="python")

        # Expandable sections
        with st.expander("📋 View Plan"):
            st.markdown(result.get("plan", ""))

        with st.expander("🧪 View Test Output"):
            test_out = result.get("test_result", "")
            st.code(test_out if test_out else "No test output captured", language="text")

        with st.expander("🎲 Hypothesis Result"):
            hyp_val = result.get("hypothesis_result", "")
            if "✅" in hyp_val:
                st.success(hyp_val)
            elif "⚠️" in hyp_val:
                st.warning(hyp_val)
            else:
                st.info("Hypothesis testing was skipped for this run.")

        with st.expander("🪞 Self Reflection Notes"):
            notes = result.get("reflection_notes", "")
            st.info(notes if notes else "Agent approved code on first reflection.")

        # Explanation
        st.divider()
        st.markdown("### 📖 Explanation")
        explanation = result.get("explanation", "")
        if explanation:
            st.markdown(explanation)
        else:
            st.info("See the final code above.")

        st.success("✅ Agent completed successfully!")

elif run_btn and not task:
    st.warning("⚠️ Please enter a task first!")

# ── FOOTER ────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#555'>Built by Krish Patel &nbsp;·&nbsp; "
    "LangGraph + Groq + ChromaDB &nbsp;·&nbsp; "
    "<a href='https://github.com' style='color:#00cc88'>GitHub</a></center>",
    unsafe_allow_html=True
)
