"""Part 2: Gemini assistant that answers ONLY by calling database tools.

CLI:  python assistant.py            (interactive)
      python assistant.py "How many rooms does plan 12 have?"
Needs GEMINI_API_KEY in the environment or in a local .env file (never committed).
"""
import functools
import os
import sys
from pathlib import Path

import tools as T

# Load .env if present
env_file = Path(__file__).with_name(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM = f"""You answer questions about a database of 121 floor plans extracted from segmentation masks.

{T.SCHEMA_TEXT.strip()}

RULES (strict):
1. Every factual statement must come from a tool result. Never answer from memory or guess.
   Use run_sql / describe_plan for every question about plans, rooms, doors or windows.
2. Areas are in PIXELS (and area_pct), never square metres. Say so when you give an area.
3. Rooms are unnamed (room_no 1..n, 1 = largest). Do not invent names like 'kitchen' or 'bedroom'.
4. If the data cannot answer the question (room names, real-world size, furniture, floors, other
   plans not in the database, advice or opinions), say clearly that the data does not contain it,
   explain what IS available, and do not make up a number. Do not call tools just to look busy.
5. If a plan id does not exist or a query returns no rows, say so.
6. Keep answers short: the answer first, then one line on how it was derived."""


def _load_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    env = Path(__file__).with_name(".env")
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        sys.exit("Set GEMINI_API_KEY (see .env.example). Free key: https://aistudio.google.com/apikey")
    return key


def _logged(fn, log):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        out = fn(*a, **kw)
        log.append({"tool": fn.__name__, "args": kw or list(a), "result": out})
        return out
    return wrapper


def ask(question: str, model: str = MODEL) -> dict:
    """Return {'answer': str, 'calls': [{'tool','args','result'}], 'grounded': bool}."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_load_key())
    calls: list = []
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=0,
        tools=[_logged(f, calls) for f in T.TOOLS],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=8),
    )
    resp = client.models.generate_content(model=model, contents=question, config=cfg)
    data_calls = [c for c in calls if c["tool"] in ("run_sql", "describe_plan")]
    return {"answer": (resp.text or "").strip(), "calls": calls, "grounded": bool(data_calls)}


def show(r: dict):
    print("\n" + r["answer"])
    if not r["grounded"]:
        print("\n[!] No database lookup was made for this answer - treat it as unverified.")
    for c in r["calls"]:
        print(f"\n  -> {c['tool']}({c['args']})")
        res = c["result"]
        print("     " + str(res)[:400])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(ask(" ".join(sys.argv[1:])))
    else:
        print("Ask about the floor plans (empty line to quit).")
        while (q := input("\n? ").strip()):
            show(ask(q))
