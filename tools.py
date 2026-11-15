"""Data-access tools that the LLM is allowed to call. Everything is read-only."""
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("floorplans.db")
MAX_ROWS = 200

SCHEMA_TEXT = """
Tables (SQLite). Units: areas are PIXELS of the original mask image, NOT square metres
(the dataset has no scale). area_pct is a room's share of the plan's total room area.
Rooms have NO names (no 'kitchen'/'bedroom'): they are numbered room_no = 1..n per plan, 1 = largest.

plans(plan_id PK, image_file, mask_file, width_px, height_px, palette,
      n_rooms, n_doors, n_windows, n_openings, total_room_area_px, wall_area_px, wall_segments)
rooms(room_id PK, plan_id, room_no, area_px, area_pct, cx, cy,        -- cx,cy: centroid in 0..1 (x right, y down)
      bbox_x0, bbox_y0, bbox_x1, bbox_y1, n_windows, n_doors)
doors(door_id PK, plan_id, kind, cx, cy, area_px, room_a, room_b)      -- room_a/room_b = room_no touched (NULL if none)
windows(window_id PK, plan_id, cx, cy, area_px, room_no)               -- room_no = room it belongs to (NULL if none found)

Not in the data: room names/types, floors, furniture, real-world dimensions, ceiling height, orientation.
"""

_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b", re.I)


def _connect():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def get_schema() -> str:
    """Return the database schema, units and what the data does NOT contain. Call this first if unsure."""
    return SCHEMA_TEXT.strip()


def run_sql(query: str) -> dict:
    """Run ONE read-only SQLite SELECT query against the floor-plan database and return the rows.

    Args:
        query: a single SELECT statement (no semicolons-separated multiple statements).
    Returns:
        dict with 'columns', 'rows' (max 200) and 'row_count', or {'error': ...}.
    """
    q = query.strip().rstrip(";")
    if not re.match(r"^(select|with)\b", q, re.I) or ";" in q or _FORBIDDEN.search(q):
        return {"error": "Only a single read-only SELECT statement is allowed."}
    try:
        with _connect() as con:
            cur = con.execute(q)
            cols = [c[0] for c in cur.description]
            rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        return {"error": f"SQL error: {e}"}
    truncated = len(rows) > MAX_ROWS
    rows = [list(r) for r in rows[:MAX_ROWS]]
    return {"columns": cols, "rows": rows, "row_count": len(rows), "truncated": truncated}


def describe_plan(plan_id: int) -> dict:
    """Return a full description of one plan: its totals and every room (area, position, windows, doors).

    Args:
        plan_id: the plan number (1-121).
    """
    plan = run_sql(f"SELECT * FROM plans WHERE plan_id = {int(plan_id)}")
    if plan.get("error") or not plan["rows"]:
        return {"error": f"No plan with id {plan_id} in the database."}
    rooms = run_sql(f"SELECT room_no, area_px, area_pct, cx, cy, n_windows, n_doors FROM rooms WHERE plan_id = {int(plan_id)} ORDER BY room_no")
    return {"plan": dict(zip(plan["columns"], plan["rows"][0])), "rooms_columns": rooms["columns"], "rooms": rooms["rows"]}


TOOLS = [get_schema, run_sql, describe_plan]
