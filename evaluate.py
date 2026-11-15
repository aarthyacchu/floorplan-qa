"""Part 3A: 20 test questions with known answers.

Gold answers are computed in plain Python from the raw table rows (NOT with the SQL the LLM
might write), so a pass means the LLM's tool use agrees with an independent calculation.

  python evaluate.py --check-gold    # no API key needed: sanity-checks gold vs. reference SQL
  python evaluate.py                 # runs the Gemini assistant on all 20, writes eval_report.md
"""
import argparse, json, re, sqlite3, time
from pathlib import Path

import tools as T

DB = Path(__file__).with_name("floorplans.db")


def load():
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    g = lambda t: [dict(r) for r in con.execute(f"SELECT * FROM {t}")]
    return g("plans"), g("rooms"), g("doors"), g("windows")


def build_questions():
    plans, rooms, doors, windows = load()
    P = {p["plan_id"]: p for p in plans}
    R = lambda pid: sorted([r for r in rooms if r["plan_id"] == pid], key=lambda r: r["room_no"])
    cnt = lambda rows, pid: sum(1 for r in rows if r["plan_id"] == pid)
    ids = sorted(P)
    q = []
    add = lambda text, kind, gold, sql=None: q.append(dict(q=text, kind=kind, gold=gold, sql=sql))

    add("How many rooms does plan 12 have?", "num", cnt(rooms, 12), "SELECT COUNT(*) FROM rooms WHERE plan_id=12")
    big7 = max(R(7), key=lambda r: r["area_px"])
    add("What is the largest room in plan 7? Give its room number.", "num", big7["room_no"],
        "SELECT room_no FROM rooms WHERE plan_id=7 ORDER BY area_px DESC LIMIT 1")
    add("Which plans have more than 15 doors?", "set", [i for i in ids if cnt(doors, i) > 15],
        "SELECT plan_id FROM plans WHERE n_doors>15")
    add("Which room in plan 4 has no windows?", "set", [r["room_no"] for r in R(4) if r["n_windows"] == 0],
        "SELECT room_no FROM rooms WHERE plan_id=4 AND n_windows=0")
    add("What is the average number of rooms across all plans?", "float", sum(p["n_rooms"] for p in plans) / len(plans),
        "SELECT AVG(n_rooms) FROM plans")
    add("How many windows does plan 1 have?", "num", cnt(windows, 1), "SELECT COUNT(*) FROM windows WHERE plan_id=1")
    add("How many doors does plan 12 have?", "num", cnt(doors, 12), "SELECT COUNT(*) FROM doors WHERE plan_id=12")
    most = max(ids, key=lambda i: (cnt(rooms, i), -i))
    add("Which plan has the most rooms?", "num", most, "SELECT plan_id FROM plans ORDER BY n_rooms DESC, plan_id LIMIT 1")
    add("How many plans are in the database?", "num", len(plans), "SELECT COUNT(*) FROM plans")
    small4 = min(R(4), key=lambda r: r["area_px"])
    add("Which is the smallest room in plan 4? Give its room number.", "num", small4["room_no"],
        "SELECT room_no FROM rooms WHERE plan_id=4 ORDER BY area_px LIMIT 1")
    add("Which plans have exactly 2 rooms?", "set", [i for i in ids if cnt(rooms, i) == 2], "SELECT plan_id FROM plans WHERE n_rooms=2")
    add("How many plans have no windows at all?", "num", sum(1 for i in ids if cnt(windows, i) == 0),
        "SELECT COUNT(*) FROM plans WHERE n_windows=0")
    add("What is the average number of doors per plan?", "float", sum(cnt(doors, i) for i in ids) / len(ids),
        "SELECT AVG(n_doors) FROM plans")
    big12 = max(R(12), key=lambda r: r["area_px"])
    add("What percentage of plan 12's total room area is its largest room?", "float", round(100 * big12["area_px"] / sum(r["area_px"] for r in R(12)), 1),
        "SELECT area_pct FROM rooms WHERE plan_id=12 ORDER BY area_px DESC LIMIT 1")
    add("Which plan has the largest total room area in pixels?", "num", max(ids, key=lambda i: P[i]["total_room_area_px"]),
        "SELECT plan_id FROM plans ORDER BY total_room_area_px DESC LIMIT 1")
    add("How many rooms are there in total across all plans?", "num", len(rooms), "SELECT COUNT(*) FROM rooms")
    add("Which rooms in plan 12 touch more than 2 doors?", "set", [r["room_no"] for r in R(12) if r["n_doors"] > 2],
        "SELECT room_no FROM rooms WHERE plan_id=12 AND n_doors>2")
    # questions the assistant must DECLINE (the data cannot answer them)
    add("What is the area of the kitchen in plan 3 in square metres?", "decline", None)
    add("Which plan has a swimming pool?", "decline", None)
    add("How many rooms does plan 500 have?", "decline", None)
    return q


DECLINE = re.compile(r"(does not|doesn't|do not|don't|cannot|can't|not (contain|include|available|in the)|no (such )?(plan|data|information|record)|isn't|not possible|unable|no plan)", re.I)


def nums(text, exclude=()):
    out = []
    for m in re.finditer(r"\d[\d,]*\.?\d*", text):
        s = m.group().replace(",", "").rstrip(".")
        try:
            v = float(s)
        except ValueError:
            continue
        if v not in exclude:
            out.append(v)
    return out


def grade(item, answer, grounded):
    k, gold = item["kind"], item["gold"]
    if k == "decline":
        m2 = re.search(r"\d+(\.\d+)?\s*(m²|m2|sq|square met)", answer, re.I)
        ok = bool(DECLINE.search(answer)) and not m2
        return ok, "declined" if ok else "did not clearly decline / invented a value"
    if not grounded:
        return False, "no database lookup made"
    # drop echoes of the plan named in the question ("plan 12") so they can't pass as the answer
    for n in re.findall(r"plan\s*#?(\d+)", item["q"], re.I):
        answer = re.sub(rf"plan\s*#?{n}\b", " ", answer, flags=re.I)
    qn = set(nums(item["q"]))
    got = nums(answer)
    if k == "num":
        ok = float(gold) in got
        return ok, "" if ok else f"expected {gold}, answer numbers {got[:6]}"
    if k == "float":
        tol = 0.06 if gold < 100 else 1
        ok = any(abs(v - gold) <= tol for v in got)
        return ok, "" if ok else f"expected ~{gold:.2f}, answer numbers {got[:6]}"
    if k == "set":
        g = {float(x) for x in gold}
        found = set(nums(answer, exclude=qn))
        if not g:
            return bool(re.search(r"\b(no|none|zero|0)\b", answer, re.I)), "expected empty set"
        missing, extra = sorted(g - found), sorted(found - g)
        ok = not missing and not extra
        return ok, "" if ok else f"missing {missing[:8]} extra {extra[:8]}"
    raise ValueError(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-gold", action="store_true")
    ap.add_argument("--delay", type=float, default=7.0, help="seconds between calls (free-tier rate limits)")
    a = ap.parse_args()
    qs = build_questions()

    if a.check_gold:
        bad = 0
        for i, it in enumerate(qs, 1):
            if it["sql"] is None:
                print(f"{i:2d}. [decline] {it['q']}"); continue
            rows = T.run_sql(it["sql"])["rows"]
            flat = [float(r[0]) for r in rows]
            gold = it["gold"]
            ok = (set(flat) == {float(x) for x in gold}) if it["kind"] == "set" else (len(flat) == 1 and abs(flat[0] - float(gold)) < 0.06)
            bad += not ok
            print(f"{i:2d}. {'OK ' if ok else 'BAD'} gold={gold if it['kind']!='set' else gold[:10]}  {it['q']}")
        print(f"\nreference-SQL vs independent gold: {len(qs) - 3 - bad}/{len(qs) - 3} agree")
        return

    import assistant
    res = []
    for i, it in enumerate(qs, 1):
        for attempt in range(4):
            try:
                r = assistant.ask(it["q"]); break
            except Exception as e:  # rate limits etc.
                print(f"  retry {attempt+1}: {str(e)[:80]}", flush=True); time.sleep(20 * (attempt + 1)); r = None
        if r is None:
            r = {"answer": "(API error)", "calls": [], "grounded": False}
        ok, why = grade(it, r["answer"], r["grounded"])
        sqls = [c["args"].get("query") if isinstance(c["args"], dict) else c["args"] for c in r["calls"] if c["tool"] == "run_sql"]
        res.append(dict(n=i, question=it["q"], kind=it["kind"], gold=it["gold"], answer=r["answer"], ok=ok, why=why, sql=sqls))
        print(f"{i:2d}. {'PASS' if ok else 'FAIL'}  {it['q']}  {why}", flush=True)
        time.sleep(a.delay)
    acc = sum(r["ok"] for r in res) / len(res)
    json.dump(res, open("eval_results.json", "w"), indent=2, default=str)
    lines = [f"# Evaluation report\n\nModel: `{assistant.MODEL}` - accuracy **{sum(r['ok'] for r in res)}/{len(res)} = {acc:.0%}**\n",
             "| # | Question | Result | Note |", "|---|---|---|---|"]
    lines += [f"| {r['n']} | {r['question']} | {'PASS' if r['ok'] else 'FAIL'} | {r['why']} |" for r in res]
    lines.append("\n## Failures (answer + SQL used)\n")
    for r in res:
        if not r["ok"]:
            lines.append(f"**Q{r['n']}** {r['question']}\n- gold: `{r['gold']}`\n- answer: {r['answer'][:400]}\n- SQL: `{r['sql']}`\n")
    Path("eval_report.md").write_text("\n".join(lines))
    print(f"\naccuracy {acc:.0%} -> eval_report.md, eval_results.json")


if __name__ == "__main__":
    main()
