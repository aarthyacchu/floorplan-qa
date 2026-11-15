"""Draw extracted rooms/doors/windows over the original plan image.

Usage: python overlay.py 4 7 12        (writes overlays/plan_<id>_overlay.png)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from extract import extract_plan

Image.MAX_IMAGE_PIXELS = None


def register(img, lab, wall_idx=2, search=500):
    """Masks are not pixel-registered to their images (offset + slight scale).
    Find scale s and shift (dx,dy) that best put the mask's WALL pixels on the drawing's ink.
    Returns (s, dx, dy) in normalised units of the image size: x_img = (x_mask*s + dx)."""
    k = search / max(img.size)
    sz = (round(img.size[0] * k), round(img.size[1] * k))
    rgba = img.convert("RGBA"); bg = Image.new("RGBA", rgba.size, "white"); bg.alpha_composite(rgba)
    g = np.asarray(bg.convert("L").resize(sz, Image.BOX))
    ink = ndi.binary_dilation(g < 150, iterations=2).astype(float)
    wall = Image.fromarray(((lab == wall_idx) * 255).astype(np.uint8)).resize(sz, Image.NEAREST)
    best = (-1, 1.0, 0, 0)
    for s in np.arange(0.80, 1.31, 0.02):
        ws = np.asarray(wall.resize((max(8, round(sz[0] * s)), max(8, round(sz[1] * s))), Image.NEAREST)) > 0
        if ws.sum() == 0:
            continue
        c = fftconvolve(ink, ws[::-1, ::-1].astype(float), mode="full") / ws.sum()
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        if c[iy, ix] > best[0]:
            best = (c[iy, ix], s, (ix - (ws.shape[1] - 1)) / sz[0], (iy - (ws.shape[0] - 1)) / sz[1])
    return best[1], best[2], best[3], best[0]


def draw(plan_id, data="data", out="overlays", max_side=1400):
    plan, rooms, doors, windows, room_lbl, (s, lab) = extract_plan(Path(data) / "Y" / f"{plan_id} (2).png")
    img = Image.open(Path(data) / "X" / f"{plan_id}.png").convert("RGB")
    h, w = room_lbl.shape
    k = max_side / max(img.size)
    size = (round(img.size[0] * k), round(img.size[1] * k))
    img = img.resize(size, Image.LANCZOS)
    s_reg, dx, dy, score = register(img, lab)
    rs = Image.fromarray(room_lbl.astype(np.uint8)).resize((round(size[0] * s_reg), round(size[1] * s_reg)), Image.NEAREST)
    rl = np.zeros((size[1], size[0]), dtype=np.uint8)
    ox, oy = round(dx * size[0]), round(dy * size[1])
    ra = np.asarray(rs)
    ys0, xs0 = max(oy, 0), max(ox, 0)
    ys1, xs1 = min(oy + ra.shape[0], size[1]), min(ox + ra.shape[1], size[0])
    rl[ys0:ys1, xs0:xs1] = ra[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    tx = lambda cx: (cx * s_reg + dx) * size[0]
    ty = lambda cy: (cy * s_reg + dy) * size[1]
    rng = np.random.default_rng(3)
    cols = rng.integers(60, 230, size=(len(rooms) + 1, 3))
    arr = np.asarray(img).astype(float)
    m = rl > 0
    arr[m] = 0.55 * arr[m] + 0.45 * cols[rl[m]]
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(16, size[0] // 55))
    except OSError:
        f = ImageFont.load_default()
    for r in rooms:
        x, y = tx(r["cx"]), ty(r["cy"])
        d.text((x, y), f"R{r['room_no']}", fill=(0, 0, 0), font=f, anchor="mm", stroke_width=3, stroke_fill=(255, 255, 255))
    for dr in doors:
        x, y = tx(dr["cx"]), ty(dr["cy"])
        d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(200, 0, 0), width=3)
    for wd in windows:
        x, y = tx(wd["cx"]), ty(wd["cy"])
        d.rectangle([x - 7, y - 7, x + 7, y + 7], outline=(0, 70, 220), width=3)
    d.rectangle([0, 0, 620, 34], fill=(255, 255, 255))
    d.text((6, 4), f"plan {plan_id}: {plan['n_rooms']} rooms  red O=door({plan['n_doors']})  blue []=window({plan['n_windows']})  reg: scale {s_reg:.2f}",
           fill=(0, 0, 0), font=ImageFont.load_default())
    Path(out).mkdir(exist_ok=True)
    p = Path(out) / f"plan_{plan_id}_overlay.png"
    img.save(p)
    return p


if __name__ == "__main__":
    for a in sys.argv[1:] or ["4", "7", "12"]:
        print(draw(int(a)))
