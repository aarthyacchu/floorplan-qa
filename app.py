"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

import assistant

st.set_page_config(page_title="Ask your floor plan", page_icon="🏠", layout="wide")
st.title("🏠 Ask your floor plan")
st.caption("Intelligent spatial analysis & instant answers for your floor plans.")

EXAMPLES = [
    "How many rooms does plan 12 have?",
    "Which plan has the most rooms?",
    "What is the largest room in plan 7?",
    "Which is the smallest room in plan 4?",
    "Which plans have more than 15 doors?",
    "Which room in plan 4 has no windows?",
    "Which rooms in plan 12 touch more than 2 doors?",
    "How many windows does plan 1 have?",
    "What is the average number of rooms across all plans?",
    "Which plan has the largest total room area in pixels?",
    "What percentage of plan 12's total room area is its largest room?",
    "What is the area of the kitchen in plan 3 in square metres?",
]
with st.sidebar:
    st.subheader("✨ Curious? Try Asking:")
    for e in EXAMPLES:
        if st.button(e, use_container_width=True):
            st.session_state["pending"] = e

if "history" not in st.session_state:
    st.session_state["history"] = []

def render_assistant_response(r):
    st.write(r["answer"])
    if not r["grounded"]:
        st.warning("No database lookup was made for this answer - treat it as unverified.")
    for c in r.get("calls", []):
        with st.expander(f"🔎 {c['tool']}"):
            if c["tool"] == "run_sql":
                query = c["args"].get("query", "") if isinstance(c["args"], dict) else str(c["args"])
                st.code(query, language="sql")
            else:
                if isinstance(c["args"], (dict, list)):
                    st.json(c["args"])
                else:
                    st.text(str(c["args"]))
            
            res = c.get("result")
            if isinstance(res, (dict, list)):
                st.json(res, expanded=False)
            else:
                st.code(str(res), language="text")

for item in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        render_assistant_response(item["r"])

q = st.chat_input("Ask about the plans...") or st.session_state.pop("pending", None)
if q:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Querying the data..."):
            try:
                r = assistant.ask(q)
            except Exception as e:  # missing key, quota, network
                r = {"answer": f"Error: {e}", "calls": [], "grounded": True}
        render_assistant_response(r)
    st.session_state["history"].append({"q": q, "r": r})
