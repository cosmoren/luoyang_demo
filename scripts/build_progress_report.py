"""Build a 2-slide progress report for the Folsom alignment story + GHI vs GHI+sky.

Generates: ~/progress_report_2026-06-03.pptx

Slide 1: Folsom base (May-26 kt baseline) -> Folsom fixed (Jun-01 kt/4000 + delta=30)
Slide 2: GHI vs GHI+sky (Jun-01, NWP off), aggregate + CSI bins + per_horizon_curve.png

Re-run any time:
    ~/micromamba/envs/luoyang/bin/python ~/projects/luoyang_demo/scripts/build_progress_report.py
"""

from __future__ import annotations

import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT_PATH = pathlib.Path("~/projects/luoyang_demo/progress_report_2026-06-03.pptx").expanduser()
PER_HORIZON_PNG = pathlib.Path(
    "~/experiments_archive/ghi_vs_ghi_sky_20ep_2026-06-01/eval_outputs/per_horizon_curve.png"
).expanduser()

assert PER_HORIZON_PNG.exists(), f"missing plot: {PER_HORIZON_PNG}"

C_INK = RGBColor(0x1F, 0x29, 0x37)
C_MUTED = RGBColor(0x6B, 0x72, 0x80)
C_ACCENT = RGBColor(0x1D, 0x4E, 0xD8)
C_GOOD = RGBColor(0x10, 0x73, 0x3E)
C_WARN = RGBColor(0xA4, 0x4F, 0x00)
C_RULE = RGBColor(0xD0, 0xD5, 0xDD)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]


def add_text(
    slide,
    text,
    left,
    top,
    width,
    height,
    *,
    size=14,
    bold=False,
    color=C_INK,
    align=PP_ALIGN.LEFT,
):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return box


def add_rule(slide, left, top, width, color=C_RULE, weight_pt=0.75):
    line = slide.shapes.add_connector(1, left, top, left + width, top)
    line.line.color.rgb = color
    line.line.width = Pt(weight_pt)
    return line


def add_metric_block(slide, left, top, width, label, value, sub=None, value_color=C_INK):
    add_text(slide, label, left, top, width, Inches(0.25), size=10, color=C_MUTED, bold=True)
    add_text(
        slide,
        value,
        left,
        top + Inches(0.25),
        width,
        Inches(0.5),
        size=22,
        bold=True,
        color=value_color,
    )
    if sub:
        add_text(
            slide,
            sub,
            left,
            top + Inches(0.78),
            width,
            Inches(0.3),
            size=10,
            color=C_MUTED,
        )


def add_results_table(slide, left, top, width, rows, *, col_aligns=None, header_tone=C_ACCENT):
    """rows[0] is the header row; subsequent rows are data."""
    n_cols = len(rows[0])
    n_rows = len(rows)
    row_h = Inches(0.32)
    height = row_h * n_rows
    tbl_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    tbl = tbl_shape.table
    for c, header in enumerate(rows[0]):
        cell = tbl.cell(0, c)
        cell.text = ""
        tf = cell.text_frame
        tf.margin_left = Emu(36000)
        tf.margin_right = Emu(36000)
        tf.margin_top = Emu(18000)
        tf.margin_bottom = Emu(18000)
        p = tf.paragraphs[0]
        if col_aligns and c < len(col_aligns):
            p.alignment = col_aligns[c]
        run = p.add_run()
        run.text = header
        run.font.bold = True
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_tone
    for r in range(1, n_rows):
        for c, txt in enumerate(rows[r]):
            cell = tbl.cell(r, c)
            cell.text = ""
            tf = cell.text_frame
            tf.margin_left = Emu(36000)
            tf.margin_right = Emu(36000)
            tf.margin_top = Emu(14000)
            tf.margin_bottom = Emu(14000)
            p = tf.paragraphs[0]
            if col_aligns and c < len(col_aligns):
                p.alignment = col_aligns[c]
            run = p.add_run()
            run.text = txt
            run.font.size = Pt(11)
            run.font.color.rgb = C_INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFA, 0xFB, 0xFC) if r % 2 == 1 else RGBColor(0xFF, 0xFF, 0xFF)
    return tbl_shape


# ---------------------------------------------------------------------------
# Slide 1 -- Folsom alignment story: base (May 26) -> fixed (Jun 1)
# ---------------------------------------------------------------------------
s1 = prs.slides.add_slide(BLANK)

add_text(
    s1,
    "Folsom -- kt-aligned baseline -> kt/4000 + Huber d=30 fix",
    Inches(0.5),
    Inches(0.35),
    Inches(12.3),
    Inches(0.5),
    size=24,
    bold=True,
)
add_text(
    s1,
    "Alignment with Luoyang's effective regime closed the sky vs no-sky gap",
    Inches(0.5),
    Inches(0.85),
    Inches(12.3),
    Inches(0.35),
    size=14,
    color=C_MUTED,
)
add_rule(s1, Inches(0.5), Inches(1.25), Inches(12.3))

COL_LEFT = Inches(0.5)
COL_RIGHT = Inches(6.95)
COL_W = Inches(5.9)
COL_TOP = Inches(1.45)

# ---- LEFT: Folsom base (May 26) ----
add_text(
    s1,
    "Folsom base -- May 26, 2026",
    COL_LEFT,
    COL_TOP,
    COL_W,
    Inches(0.35),
    size=16,
    bold=True,
)
add_text(
    s1,
    "folsom_kt_sky_vs_nosky_40ep_4runs_2026-05-26  |  40 ep  |  4 runs (2x2)  |  --use-nwp",
    COL_LEFT,
    COL_TOP + Inches(0.32),
    COL_W,
    Inches(0.3),
    size=11,
    color=C_MUTED,
)

add_results_table(
    s1,
    COL_LEFT,
    COL_TOP + Inches(0.75),
    COL_W,
    rows=[
        ["arm", "test MAE (W/m^2)", "test RMSE (W/m^2)", "best ep"],
        ["PV + NWP + sky", "25.85", "69.27", "27, 15"],
        ["PV + NWP, no sky", "28.00", "72.78", "22, 20"],
        ["delta (sky helps)", "-2.15", "-3.51", "--"],
    ],
    col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
)

add_text(
    s1,
    "Takeaway",
    COL_LEFT,
    COL_TOP + Inches(2.45),
    COL_W,
    Inches(0.3),
    size=12,
    bold=True,
    color=C_MUTED,
)
add_text(
    s1,
    "Sky helps on the first-16-step horizon: -2.2 W/m^2 MAE, -3.5 W/m^2 RMSE. "
    "Intra-arm spread ~3% MAE (sky), ~8% MAE (no-sky).",
    COL_LEFT,
    COL_TOP + Inches(2.75),
    COL_W,
    Inches(1.0),
    size=12,
)

# ---- RIGHT: Folsom fixed (Jun 1) ----
add_text(
    s1,
    "Folsom fixed -- Jun 1, 2026  (commit 518dca9)",
    COL_RIGHT,
    COL_TOP,
    COL_W,
    Inches(0.35),
    size=16,
    bold=True,
    color=C_ACCENT,
)
add_text(
    s1,
    "folsom_kt4000_d30_20ep_2026-06-01  |  20 ep  |  8 runs (2 arms x 4 reps)  |  --use-nwp",
    COL_RIGHT,
    COL_TOP + Inches(0.32),
    COL_W,
    Inches(0.3),
    size=11,
    color=C_MUTED,
)

add_results_table(
    s1,
    COL_RIGHT,
    COL_TOP + Inches(0.75),
    COL_W,
    rows=[
        ["arm", "test MAE (W/m^2)", "test RMSE (W/m^2)", "n reps"],
        ["PV + NWP + sky", "25.60 +/- 0.96", "72.03 +/- 1.55", "4"],
        ["PV + NWP, no sky", "25.13 +/- 0.81", "73.47 +/- 2.14", "4"],
        ["delta (sky helps)", "+0.47 (tied)", "-1.44", "--"],
    ],
    col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
    header_tone=C_ACCENT,
)

add_text(
    s1,
    "Code shipped in 518dca9",
    COL_RIGHT,
    COL_TOP + Inches(2.45),
    COL_W,
    Inches(0.3),
    size=12,
    bold=True,
    color=C_MUTED,
)
add_text(
    s1,
    "- ViT input kt/20 -> kt/4000; train/eval output rescale 20 -> 4000\n"
    "- HuberLoss delta 1.0 -> 30.0\n"
    "- GHI scale 1100 -> 1000;  p_mean_scalar = 1.0",
    COL_RIGHT,
    COL_TOP + Inches(2.75),
    COL_W,
    Inches(1.4),
    size=11,
)

# ---- Bottom verdict band ----
add_rule(s1, Inches(0.5), Inches(6.45), Inches(12.3))
add_text(
    s1,
    "Verdict",
    Inches(0.5),
    Inches(6.55),
    Inches(1.5),
    Inches(0.3),
    size=12,
    bold=True,
    color=C_GOOD,
)
add_text(
    s1,
    "Alignment lifted the no-sky arm by 10% MAE (28.00 -> 25.13). Sky-vs-no-sky gap collapsed from ~2 W/m^2 to ~0. "
    "Open follow-up: the sky branch may now be effectively muted -- worth an isolation test (next slide).",
    Inches(2.0),
    Inches(6.55),
    Inches(10.8),
    Inches(0.9),
    size=12,
)

# ---------------------------------------------------------------------------
# Slide 2 -- GHI vs GHI+sky (no NWP), aggregate + CSI bins + per_horizon plot
# ---------------------------------------------------------------------------
s2 = prs.slides.add_slide(BLANK)

add_text(
    s2,
    "Folsom -- sky branch isolation: GHI vs GHI+sky (NWP off)",
    Inches(0.5),
    Inches(0.35),
    Inches(12.3),
    Inches(0.5),
    size=24,
    bold=True,
)
add_text(
    s2,
    "ghi_vs_ghi_sky_20ep_2026-06-01  |  6 runs (3+3)  |  20 ep  |  --zero-sky vs default  |  NWP off in both arms",
    Inches(0.5),
    Inches(0.85),
    Inches(12.3),
    Inches(0.35),
    size=13,
    color=C_MUTED,
)
add_rule(s2, Inches(0.5), Inches(1.25), Inches(12.3))

# LEFT column: aggregate + CSI summary
L_TOP = Inches(1.45)
L_LEFT = Inches(0.5)
L_W = Inches(6.2)

add_text(
    s2,
    "Aggregate -- first-16-step test set",
    L_LEFT,
    L_TOP,
    L_W,
    Inches(0.35),
    size=15,
    bold=True,
)
add_results_table(
    s2,
    L_LEFT,
    L_TOP + Inches(0.4),
    L_W,
    rows=[
        ["arm (mean of 3 seeds)", "RMSE", "MAE", "delta vs GHI-only"],
        ["GHI-only", "77.35", "28.16", "--"],
        ["GHI + sky (all 3)", "75.13", "26.19", "-2.9% / -7.0%"],
        ["GHI + sky (drop gpu2 stall)", "72.76", "24.86", "-5.9% / -11.7%"],
    ],
    col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
)

add_text(
    s2,
    "CSI regime slicing -- input-CSI bins (no-gpu2 sky line vs GHI-only)",
    L_LEFT,
    L_TOP + Inches(1.95),
    L_W,
    Inches(0.35),
    size=15,
    bold=True,
)
add_results_table(
    s2,
    L_LEFT,
    L_TOP + Inches(2.35),
    L_W,
    rows=[
        ["bin (% of test)", "RMSE delta", "MAE delta", "note"],
        ["clear  (CSI > 0.85, 73%)", "-0.8%", "-9.5%", "near tie"],
        ["partly cloudy  (25%)", "-12.0%", "-15.4%", "sky earns its keep"],
        ["overcast  (3%, n=8)", "+0.8%", "+2.8%", "noise"],
    ],
    col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT],
)

add_text(
    s2,
    "Punchline",
    L_LEFT,
    L_TOP + Inches(4.05),
    L_W,
    Inches(0.3),
    size=12,
    bold=True,
    color=C_GOOD,
)
add_text(
    s2,
    "The aggregate table understates sky's contribution: ~73% of test windows are clear days "
    "where sky is roughly break-even by design. On the cloudy ~25%, sky cuts error by 12-15%.",
    L_LEFT,
    L_TOP + Inches(4.35),
    L_W,
    Inches(1.3),
    size=12,
)

# RIGHT column: per-horizon plot
R_LEFT = Inches(6.95)
R_W = Inches(6.0)

add_text(
    s2,
    "Per-horizon RMSE / MAE (15 min -> 4 h)",
    R_LEFT,
    L_TOP,
    R_W,
    Inches(0.35),
    size=15,
    bold=True,
)
s2.shapes.add_picture(
    str(PER_HORIZON_PNG),
    R_LEFT,
    L_TOP + Inches(0.45),
    width=R_W,
)
add_text(
    s2,
    "Sky helps at 14 of 16 horizons (slight inversion at 3.5-3.75h). "
    "Improvement shrinks with horizon: -8.8% RMSE @ 15min -> -3.3% @ 4h, as expected "
    "(clouds leave the camera FOV beyond ~30 min).",
    R_LEFT,
    Inches(6.05),
    R_W,
    Inches(1.3),
    size=11,
    color=C_MUTED,
)

prs.save(OUT_PATH)
print(f"wrote {OUT_PATH}")
