"""Part 1: turn floor-plan masks into a structured SQLite database.

Usage:  python extract.py --data data --db floorplans.db

Mask colours (found by inspecting all 121 masks) come in TWO palettes:
  palette "orange_rooms": rooms=orange  walls=light-blue  windows=yellow  doors=green
  palette "teal_rooms"  : rooms=teal    walls=light-blue  windows=yellow  doors=orange,
                          green=large opening (garage/sliding door), red=thin opening
The palette is chosen per plan: whichever of orange/teal covers more area is "room".
Rooms all share one colour, so each room is a connected component of that colour
(walls/doors leave gaps between rooms).
"""
import argparse, json, sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None
WORK = 1000  # masks are analysed at max side = 1000 px; areas are rescaled to original px

COLORS = {  # reference RGB values seen in the masks
    "bg": (0, 0, 0), "orange": (255, 142, 70), "lblue": (175, 216, 248),
    "yellow": (246, 189, 15), "green": (139, 186, 0), "teal": (0, 142, 142),
    "red": (214, 70, 70),
}
NAMES = list(COLORS)
PAL = np.array([COLORS[n] for n in NAMES], dtype=np.int32)

SCHEMA = """
DROP TABLE IF EXISTS plans; DROP TABLE IF EXISTS rooms;
DROP TABLE IF EXISTS doors; DROP TABLE IF EXISTS windows;
CREATE TABLE plans(
  plan_id INTEGER PRIMARY KEY, image_file TEXT, mask_file TEXT,
  width_px INT, height_px INT, palette TEXT,
  n_rooms INT, n_doors INT, n_windows INT, n_openings INT,
  total_room_area_px INT, wall_area_px INT, wall_segments INT);
CREATE TABLE rooms(
  room_id INTEGER PRIMARY KEY, plan_id INT, room_no INT,  -- room_no is 1..n within plan, ordered by area desc
  area_px INT, area_pct REAL,              -- area_pct = share of total room area in plan
  cx REAL, cy REAL,                        -- centroid, normalised 0..1 (x right, y down)
  bbox_x0 INT, bbox_y0 INT, bbox_x1 INT, bbox_y1 INT,  -- original-pixel bounding box
  n_windows INT, n_doors INT);
CREATE TABLE doors(
  door_id INTEGER PRIMARY KEY, plan_id INT, kind TEXT,  -- 'door'
  cx REAL, cy REAL, area_px INT,
  room_a INT, room_b INT);                 -- room_no of rooms touched (NULL = none/exterior)
CREATE TABLE windows(
  window_id INTEGER PRIMARY KEY, plan_id INT,
  cx REAL, cy REAL, area_px INT, room_no INT);  -- room_no it belongs to (NULL if none found)
CREATE INDEX i_r ON rooms(plan_id); CREATE INDEX i_d ON doors(plan_id); CREATE INDEX i_w ON windows(plan_id);
"""


def classify(mask_path):
    """Return (label array of palette indices at reduced size, scale factor, (W,H))."""
    im = Image.open(mask_path).convert("RGB")
    W, H = im.size
    s = max(1, max(W, H) / WORK)
    small = im.resize((round(W / s), round(H / s)), Image.NEAREST)
    a = np.asarray(small, dtype=np.int32)
    d = ((a[:, :, None, :] - PAL[None, None]) ** 2).sum(-1)
    lab = d.argmin(-1).astype(np.uint8)
    lab[d.min(-1) > 60 ** 2] = 0  # anti-aliasing / unknown colours -> background
    return lab, s, (W, H)


def components(binary, min_px):
    lbl, n = ndi.label(binary)
    if n == 0:
        return lbl, []
    sizes = ndi.sum(binary, lbl, range(1, n + 1))
    keep = [i + 1 for i, sz in enumerate(sizes) if sz >= min_px]
    return lbl, keep


def touching_rooms(comp_mask, room_lbl, radius):
    """Room labels within `radius` px of a component (needs >=3 overlapping px)."""
    ys, xs = np.where(comp_mask)
    y0, y1 = max(ys.min() - radius - 1, 0), ys.max() + radius + 2
    x0, x1 = max(xs.min() - radius - 1, 0), xs.max() + radius + 2
    sub = comp_mask[y0:y1, x0:x1]
    dil = ndi.binary_dilation(sub, iterations=radius)
    labs, cnt = np.unique(room_lbl[y0:y1, x0:x1][dil], return_counts=True)
    return [int(l) for l, c in zip(labs, cnt) if l > 0 and c >= 3]


def extract_plan(mask_path):
    lab, s, (W, H) = classify(mask_path)
    h, w = lab.shape
    count = lambda name: int((lab == NAMES.index(name)).sum())
    orange, teal = count("orange"), count("teal")
    if teal > orange:
        palette, room_c, door_cs = "teal_rooms", "teal", ["orange", "green"]
    else:
        palette, room_c, door_cs = "orange_rooms", "orange", ["green"]
    opening_cs = ["red"] if palette == "teal_rooms" else []

    # rooms: connected components of the room colour, ignoring tiny specks
    room_bin = lab == NAMES.index(room_c)
    min_room = max(150, int(0.0008 * h * w))
    rl, keep = components(room_bin, min_room)
    keep = sorted(keep, key=lambda i: -(rl == i).sum())
    room_lbl = np.zeros_like(rl)
    for no, i in enumerate(keep, 1):
        room_lbl[rl == i] = no
    area_scale = (W * H) / (w * h)
    total_room_px = int(room_bin[np.isin(rl, keep)].sum() * area_scale)

    rooms = []
    for no in range(1, len(keep) + 1):
        m = room_lbl == no
        ys, xs = np.where(m)
        rooms.append(dict(
            room_no=no, area_px=int(m.sum() * area_scale),
            cx=float(xs.mean() / w), cy=float(ys.mean() / h),
            bbox=[int(xs.min() * s), int(ys.min() * s), int((xs.max() + 1) * s), int((ys.max() + 1) * s)],
            n_windows=0, n_doors=0))

    def ents(colors, min_px, radius):
        binary = np.isin(lab, [NAMES.index(c) for c in colors])
        el, ek = components(binary, min_px)
        out = []
        for i in ek:
            m = el == i
            ys, xs = np.where(m)
            out.append(dict(cx=float(xs.mean() / w), cy=float(ys.mean() / h),
                            area_px=int(m.sum() * area_scale),
                            rooms=touching_rooms(m, room_lbl, radius)))
        return out

    min_e = max(20, int(0.00003 * h * w))
    doors = ents(door_cs, min_e, 8)
    windows = ents(["yellow"], min_e, 16)
    openings = ents(opening_cs, min_e, 8) if opening_cs else []

    d_rows = []
    for d in doors:
        r = d["rooms"][:2]
        d_rows.append(dict(cx=d["cx"], cy=d["cy"], area_px=d["area_px"],
                           room_a=r[0] if len(r) > 0 else None, room_b=r[1] if len(r) > 1 else None))
        for no in d["rooms"]:
            rooms[no - 1]["n_doors"] += 1
    w_rows = []
    for wd in windows:
        # a window belongs to the room it overlaps most closely; pick the first (largest) touching
        no = wd["rooms"][0] if wd["rooms"] else None
        w_rows.append(dict(cx=wd["cx"], cy=wd["cy"], area_px=wd["area_px"], room_no=no))
        for no2 in wd["rooms"]:
            rooms[no2 - 1]["n_windows"] += 1
    wall_bin = lab == NAMES.index("lblue")
    wall_seg = components(wall_bin, min_e)[1]
    plan = dict(width_px=W, height_px=H, palette=palette, n_rooms=len(rooms), n_doors=len(d_rows),
                n_windows=len(w_rows), n_openings=len(openings), total_room_area_px=total_room_px,
                wall_area_px=int(wall_bin.sum() * area_scale), wall_segments=len(wall_seg))
    return plan, rooms, d_rows, w_rows, room_lbl, (s, lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--db", default="floorplans.db")
    a = ap.parse_args()
    data = Path(a.data)
    con = sqlite3.connect(a.db)
    con.executescript(SCHEMA)
    ids = sorted(int(p.stem) for p in (data / "X").glob("*.png"))
    for pid in ids:
        mp = data / "Y" / f"{pid} (2).png"
        if not mp.exists():
            print(f"plan {pid}: no mask, skipped"); continue
        plan, rooms, doors, windows, *_ = extract_plan(mp)
        con.execute("INSERT INTO plans VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, f"X/{pid}.png", f"Y/{pid} (2).png", plan["width_px"], plan["height_px"],
                     plan["palette"], plan["n_rooms"], plan["n_doors"], plan["n_windows"],
                     plan["n_openings"], plan["total_room_area_px"], plan["wall_area_px"],
                     plan["wall_segments"]))
        tot = max(plan["total_room_area_px"], 1)
        for r in rooms:
            con.execute("INSERT INTO rooms(plan_id,room_no,area_px,area_pct,cx,cy,bbox_x0,bbox_y0,bbox_x1,bbox_y1,n_windows,n_doors) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (pid, r["room_no"], r["area_px"], round(100 * r["area_px"] / tot, 2), r["cx"], r["cy"], *r["bbox"], r["n_windows"], r["n_doors"]))
        for d in doors:
            con.execute("INSERT INTO doors(plan_id,kind,cx,cy,area_px,room_a,room_b) VALUES(?,?,?,?,?,?,?)",
                        (pid, "door", d["cx"], d["cy"], d["area_px"], d["room_a"], d["room_b"]))
        for wd in windows:
            con.execute("INSERT INTO windows(plan_id,cx,cy,area_px,room_no) VALUES(?,?,?,?,?)",
                        (pid, wd["cx"], wd["cy"], wd["area_px"], wd["room_no"]))
        print(f"plan {pid:3d} [{plan['palette']:12s}] rooms={plan['n_rooms']:2d} doors={plan['n_doors']:2d} windows={plan['n_windows']:2d}")
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
