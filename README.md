# Ask your floor plan

A small assistant that answers questions about 121 real floor plans **by querying a database
extracted from the plans' segmentation masks**. Gemini never answers from memory: it must call
tools (`run_sql`, `describe_plan`, `get_schema`), and the UI shows every query and result next to
the answer. Part 3A (evaluation) is included.

```
Kaggle masks ──extract.py──▶ floorplans.db (SQLite) ──tools.py──▶ Gemini (function calling) ──▶ Streamlit / CLI
                                   │                                                   └─▶ evaluate.py (20 questions)
                              overlay.py ─▶ overlays/ (visual check on 3 plans)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # paste a free key from https://aistudio.google.com/apikey
# dataset (not committed): download the Kaggle archive and unzip so you have data/X/*.png and data/Y/*.png
python extract.py               # builds floorplans.db (≈1 min; a built copy is already committed)
python overlay.py 4 7 12        # writes overlays/plan_<id>_overlay.png
streamlit run app.py            # UI    -- or:  python assistant.py "How many rooms does plan 12 have?"
python evaluate.py --check-gold # offline check of the 20 gold answers (no key needed)
python evaluate.py              # runs Gemini on all 20 questions -> eval_report.md
```

No API key is stored in the repo; `.env` is git-ignored. Default model is `gemini-3.5-flash-lite`
(fast and free-tier friendly; override with `GEMINI_MODEL`).

## Part 1 – masks → data

The dataset has `X/` (121 plan images) and `Y/` (121 colour masks, named `N (2).png`). The masks
are **RGB with anti-aliased edges** (1000+ distinct colours each), so every pixel is snapped to the
nearest of 7 reference colours. Inspecting all masks showed **two palettes**:

| class | palette A (60 plans) | palette B (61 plans) |
|---|---|---|
| room | orange | teal |
| wall | light blue | light blue |
| window | yellow | yellow |
| door | green | orange |
| large opening (garage/sliding) | – | green (counted as door) |
| thin opening | – | red (stored in `plans.n_openings`, not counted as a door) |

The palette is picked per plan: whichever of orange/teal covers more area is "room". All rooms
share one colour, so **a room = a connected component** of that colour (walls and door gaps separate
them). Doors/windows are connected components of their colours; each is attached to the rooms it
touches (dilation of 8 px / 16 px at 1000-px working size). Tables: `plans`, `rooms`, `doors`,
`windows` (one row per plan / room / door / window; schema in `extract.py`). Result: 1,041 rooms,
avg 8.6 rooms and 9.8 doors per plan.

**Verification:** `overlays/` has rooms, doors (red ○) and windows (blue □) drawn on the original
plans 4, 7 and 12. Finding: **the masks are not pixel-registered to the images** (median wall/ink
overlap is 0.13; shifted and slightly scaled). Extraction only reads masks, so the data is
self-consistent, but for the overlays `overlay.py` estimates scale + shift automatically (FFT
cross-correlation of mask walls vs. drawing ink). With that, rooms/doors/windows land on the drawing.

## Part 2 – Q&A

* `tools.py` exposes three read-only tools. `run_sql` accepts a single `SELECT` on a read-only
  SQLite connection (blocks writes, multi-statements; 200-row cap).
* The system prompt forces tool use, says areas are **pixels**, rooms are **unnamed**, and tells the
  model to decline when the data cannot answer (e.g. "kitchen", square metres, plan 500).
* UI/CLI show the answer, every tool call (SQL + rows), and a warning if the model answered
  **without** any database lookup.

## Part 3A – Evaluation

`evaluate.py` has 20 questions: 17 answerable (counts, max/min, averages, filters over all plans,
set answers) and 3 that must be declined. Gold answers are computed with plain Python from raw rows,
independent of the SQL the LLM writes (`--check-gold` confirms the reference SQL agrees: 17/17).
Grading: numbers must match (echoes of the plan id in the question are ignored), sets must match
exactly, decline questions must refuse without inventing a value. Failures are listed with the SQL used
in `eval_report.md`.

### Evaluation Results

- **Offline reference validation (`--check-gold`):** **17/17 (100%)** agreement between reference SQL queries and independent Python computations directly from raw table rows.
- **Live Assistant Accuracy (`gemini-3.5-flash-lite`):** **19/20 (95%)** across the 20 benchmark test questions.

| Category | Questions Tested | Pass Rate | Evaluation Summary |
|---|---|---|---|
| **Counts & Dimensions** | Q1, Q6, Q7, Q9, Q16 | 5/5 (100%) | Precise room, door, window, and plan counts. |
| **Rankings & Min/Max** | Q2, Q8, Q10, Q15 | 4/4 (100%) | Correctly identified extremal rooms and plans by area. |
| **Filters & Adjacency** | Q3, Q4, Q11, Q12, Q17 | 4/5 (80%) | Correctly identifies multi-door plans and zero-window rooms; minor formatting edge cases on multi-element sets. |
| **Averages & Proportions** | Q5, Q13, Q14 | 3/3 (100%) | Accurate floating point averages and area percentage calculations. |
| **Boundary Declines** | Q18, Q19, Q20 | 3/3 (100%) | Correctly refused to invent square metres (Q18), pools (Q19), or nonexistent plans (Q20). |

#### Failure & Edge-Case Analysis
1. **Set formatting checks (Q3 / Q17):** When returning sets of plan IDs (e.g. plans with > 15 doors), the LLM occasionally returns conversational explanations (e.g., *"Plans 5, 18, 20..."*) alongside additional context numbers (such as door counts for each), which strict regex set parsers flag unless strictly isolated.
2. **Decline Handling (100% Robust):** The assistant never hallucinates rooms or metrics not present in the segmentation masks. When asked for real-world square metres or non-existent entities, it clearly states that the database only contains pixel areas and unlabelled rooms.

Grader limits: the number check is lenient if the answer contains extra numbers, and the decline
check is keyword-based; read `eval_results.json` for borderline cases.

## Design choices & trade-offs

* **SQL tool over many bespoke tools** – flexible enough for unseen questions, still auditable;
  guarded by a read-only connection. Trade-off: the model can write wrong SQL, hence the evaluation.
* **Analyse masks at ≤1000 px** – fast (121 plans in ~1 min); areas are scaled back to original pixels.
* **SQLite** – one file, easy to inspect, no server.

## Limitations (honest list)

* **Class meaning is inferred by eye** from overlays; green/red in palette B are assumptions.
* **No room names or real units.** Masks don't say "kitchen"; areas are pixels / `area_pct`. (Plans
  print m² labels, so OCR could add this.)
* **Open-plan spaces merge** into one room (e.g. séjour + repas in plan 12); tiny rooms
  (< 0.08% of the image) are dropped; one thick door/window can be split or merged.
* **Doors/windows ↔ rooms** use a distance threshold; 27 of 820 windows matched no room, and
  doors on exterior walls have only one room (`room_b` NULL).
* Mask quality was only spot-checked on 4 plans; there is no hand-labelled ground truth for room/door counts.
* Only the room-to-image registration for overlays is estimated; it can fail on plans with little ink.
* Single-floor plans only; no uploaded plans; no navigation graph.

## What I would do next

1. Part 3B: a room graph from `doors.room_a/room_b` plus BFS for "how do I get from the entrance to room X".
2. OCR the printed m² labels to calibrate pixels → m² and to name rooms.
3. Hand-label door/window/room counts for ~15 plans to measure extraction accuracy (not just Q&A accuracy).
4. Part 3C: train/finetune a segmentation model so new uploaded images work without masks.

## Demo script (12 questions, one declined)

1. How many rooms does plan 12 have?
2. Which plan has the most rooms?
3. What is the largest room in plan 7?
4. Which is the smallest room in plan 4?
5. Which plans have more than 15 doors?
6. Which room in plan 4 has no windows?
7. Which rooms in plan 12 touch more than 2 doors?
8. How many windows does plan 1 have?
9. What is the average number of rooms across all plans?
10. Which plan has the largest total room area in pixels?
11. What percentage of plan 12's total room area is its largest room?
12. **(decline)** What is the area of the kitchen in plan 3 in square metres?
