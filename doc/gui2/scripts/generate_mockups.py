#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mockups"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "bg_top": (243, 247, 252),
    "bg_bottom": (229, 238, 245),
    "surface": (255, 255, 255),
    "surface_alt": (244, 249, 252),
    "ink": (23, 35, 49),
    "muted": (88, 102, 120),
    "line": (206, 219, 230),
    "primary": (21, 128, 141),
    "primary_soft": (220, 242, 245),
    "accent": (201, 111, 45),
    "accent_soft": (248, 232, 219),
    "ok": (44, 140, 85),
    "warn": (182, 128, 39),
    "error": (182, 69, 69),
}

FONT_PATHS = {
    "regular": "/usr/share/fonts/inriafonts/InriaSans-Regular.otf",
    "bold": "/usr/share/fonts/inriafonts/InriaSans-Bold.otf",
    "serif": "/usr/share/fonts/inriafonts/InriaSerif-Bold.otf",
}

LAYOUT_1920 = {
    "viewport_w": 1920,
    "viewport_h": 1080,
    "outer_margin": 24,
    "topbar": (24, 20, 1896, 96),
    "workspace": (24, 116, 1896, 1056),
    "sidebar": (24, 116, 324, 1056),
    "main_area": (356, 116, 1894, 1056),
    "main_wrapper": (356, 230, 1894, 1028),
    "dashboard_guided": (356, 402, 1210, 860),
    "dashboard_guard": (1230, 402, 1894, 860),
    "parameter_cols": ((382, 346, 732, 996), (748, 346, 1380, 996), (1398, 346, 1868, 996)),
    "run_cols": ((382, 358, 1030, 996), (1048, 358, 1468, 996), (1486, 358, 1868, 996)),
    "history_cols": ((382, 266, 1242, 996), (1260, 266, 1868, 648), (1260, 666, 1868, 996)),
}

SPACING_1920 = {
    "global": {
        "wrapper_inner_padding": 24,
        "field_label_to_input_gap": 10,
    },
    "dashboard": {
        "readiness_first_row_y": 520,
        "readiness_row_h": 60,
        "readiness_row_step": 80,
    },
    "parameter_studio": {
        "registration_row1_y": 424,
        "registration_row2_y": 516,
        "registration_row2_hgap": 20,
        "section_title_gap": 34,
    },
    "run_monitor": {
        "artifact_list_first_y": 480,
        "artifact_button_row_y": 804,
        "artifact_secondary_button_y": 868,
    },
    "history_tools": {
        "astrometry_first_input_y": 356,
        "astrometry_row_step": 86,
        "astrometry_plate_solve_y": 528,
    },
}


def load_font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_PATHS[kind]
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.load_default()


def gradient_background(img: Image.Image, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Subtle abstract shapes.
    for box, color in [
        ((-180, -120, 520, 440), (224, 236, 248)),
        ((w - 520, -180, w + 100, 420), (230, 243, 236)),
        ((w - 620, h - 380, w + 60, h + 120), (238, 231, 248)),
        ((-140, h - 260, 520, h + 220), (247, 236, 226)),
    ]:
        draw.ellipse(box, fill=color)


def rr(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill, outline=None, width: int = 1, radius: int = 18):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, x: int, y: int, s: str, *, font, fill, anchor: str = "la"):
    draw.text((x, y), s, font=font, fill=fill, anchor=anchor)


def wrap_text(draw: ImageDraw.ImageDraw, msg: str, font, max_width: int) -> list[str]:
    words = msg.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = w if not cur else f"{cur} {w}"
        bbox = draw.textbbox((0, 0), cand, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_paragraph(draw: ImageDraw.ImageDraw, x: int, y: int, msg: str, *, font, fill, max_width: int, line_height: int) -> int:
    lines = wrap_text(draw, msg, font, max_width)
    yy = y
    for line in lines:
        text(draw, x, yy, line, font=font, fill=fill)
        yy += line_height
    return yy


def button(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str, *, primary: bool = False):
    fill = PALETTE["primary"] if primary else PALETTE["surface"]
    outline = None if primary else PALETTE["line"]
    rr(draw, (x, y, x + w, y + h), fill=fill, outline=outline, width=1, radius=12)
    # Slightly reduce font for longer labels to avoid visual crowding/overflow.
    font_size = 20 if len(label) >= 16 else 22
    f = load_font("bold", font_size)
    tcol = (250, 253, 255) if primary else PALETTE["ink"]
    text(draw, x + w // 2, y + h // 2 + 1, label, font=f, fill=tcol, anchor="mm")


def chip(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, *, kind: str = "neutral"):
    styles = {
        "neutral": (PALETTE["surface_alt"], PALETTE["muted"]),
        "ok": ((223, 243, 232), PALETTE["ok"]),
        "warn": ((248, 236, 214), PALETTE["warn"]),
        "error": ((247, 222, 222), PALETTE["error"]),
        "primary": (PALETTE["primary_soft"], PALETTE["primary"]),
    }
    bg, fg = styles[kind]
    f = load_font("bold", 17)
    bbox = draw.textbbox((0, 0), label, font=f)
    tw = bbox[2] - bbox[0]
    ww = tw + 28
    hh = 34
    rr(draw, (x, y, x + ww, y + hh), fill=bg, outline=None, radius=17)
    text(draw, x + ww // 2, y + hh // 2 + 1, label, font=f, fill=fg, anchor="mm")


def input_field(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str, value: str, *, mono: bool = False):
    lf = load_font("bold", 16)
    vf = load_font("regular", 18)
    label_gap = SPACING_1920["global"]["field_label_to_input_gap"]
    text(draw, x, y - label_gap, label, font=lf, fill=PALETTE["muted"], anchor="lb")
    rr(draw, (x, y, x + w, y + h), fill=PALETTE["surface"], outline=PALETTE["line"], width=1, radius=10)
    text(draw, x + 14, y + h // 2 + 1, value, font=vf, fill=(64, 78, 95), anchor="lm")


def metric_card(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, value: str, delta: str, *, trend: str = "ok"):
    rr(draw, (x, y, x + w, y + h), fill=PALETTE["surface"], outline=PALETTE["line"], width=1, radius=16)
    text(draw, x + 18, y + 28, title, font=load_font("bold", 18), fill=PALETTE["muted"], anchor="lm")
    text(draw, x + 18, y + 70, value, font=load_font("serif", 40), fill=PALETTE["ink"], anchor="lm")
    chip(draw, x + 18, y + 92, delta, kind=trend)


def split_row(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    segments: list[tuple[int, str, tuple[int, int, int]]],
    *,
    gaps: list[int] | None = None,
):
    gaps = gaps or []
    total_gap = sum(gaps)
    total_segments = sum(seg_w for seg_w, _, _ in segments)
    scale = (w - total_gap) / max(1, total_segments)

    text(draw, x, y + h // 2 + 1, label, font=load_font("bold", 20), fill=PALETTE["ink"], anchor="lm")
    cx = x + 280
    for idx, (seg_w, seg_label, seg_color) in enumerate(segments):
        ww = int(seg_w * scale)
        rr(draw, (cx, y, cx + ww, y + h), fill=seg_color, outline=(188, 210, 226), width=1, radius=8)
        text(draw, cx + ww // 2, y + h // 2 + 1, seg_label, font=load_font("bold", 16), fill=PALETTE["ink"], anchor="mm")
        cx += ww
        if idx < len(gaps):
            gap = gaps[idx]
            draw.line((cx + gap // 2, y + 6, cx + gap // 2, y + h - 6), fill=(168, 186, 200), width=2)
            cx += gap


def h_measure(draw: ImageDraw.ImageDraw, x1: int, x2: int, y: int, label: str, color=(112, 136, 156)):
    draw.line((x1, y, x2, y), fill=color, width=2)
    draw.polygon([(x1, y), (x1 + 9, y - 6), (x1 + 9, y + 6)], fill=color)
    draw.polygon([(x2, y), (x2 - 9, y - 6), (x2 - 9, y + 6)], fill=color)
    tw = draw.textbbox((0, 0), label, font=load_font("bold", 14))[2]
    cx = (x1 + x2) // 2
    rr(draw, (cx - tw // 2 - 10, y - 24, cx + tw // 2 + 10, y - 6), fill=(255, 255, 255), outline=(204, 218, 230), radius=8)
    text(draw, cx, y - 15, label, font=load_font("bold", 14), fill=PALETTE["muted"], anchor="mm")


def draw_measure_row(
    draw: ImageDraw.ImageDraw,
    y: int,
    title: str,
    segments: list[tuple[str, int, tuple[int, int, int]]],
    *,
    start_offset: int = 0,
    gaps: list[int] | None = None,
    notes: str = "",
):
    gaps = gaps or []
    card = (56, y, 1864, y + 186)
    rr(draw, card, fill=PALETTE["surface"], outline=PALETTE["line"], radius=14)
    text(draw, 82, y + 30, title, font=load_font("serif", 30), fill=PALETTE["ink"], anchor="lm")

    wrapper_x = 260
    wrapper_y = y + 66
    wrapper_w = 1538
    wrapper_h = 58
    rr(draw, (wrapper_x, wrapper_y, wrapper_x + wrapper_w, wrapper_y + wrapper_h), fill=(248, 252, 255), outline=(158, 182, 202), width=2, radius=10)
    text(draw, wrapper_x + 12, wrapper_y + 18, "Main Wrapper 1538 px", font=load_font("bold", 13), fill=PALETTE["muted"], anchor="lm")

    cx = wrapper_x + start_offset
    seg_top = wrapper_y + 18
    seg_h = 30
    for idx, (label, width, color) in enumerate(segments):
        rr(draw, (cx, seg_top, cx + width, seg_top + seg_h), fill=color, outline=(166, 188, 206), width=1, radius=7)
        text(draw, cx + width // 2, seg_top + seg_h // 2 + 1, label, font=load_font("bold", 13), fill=PALETTE["ink"], anchor="mm")
        h_measure(draw, cx, cx + width, seg_top - 12, f"{width} px")
        cx += width
        if idx < len(gaps):
            gap = gaps[idx]
            draw.line((cx + gap // 2, wrapper_y + 8, cx + gap // 2, wrapper_y + wrapper_h - 8), fill=(176, 194, 210), width=2)
            h_measure(draw, cx, cx + gap, wrapper_y + wrapper_h + 18, f"{gap} px")
            cx += gap

    if start_offset > 0:
        h_measure(draw, wrapper_x, wrapper_x + start_offset, wrapper_y + wrapper_h + 40, f"{start_offset} px inset")
        h_measure(draw, cx, wrapper_x + wrapper_w, wrapper_y + wrapper_h + 40, f"{start_offset} px inset")

    h_measure(draw, wrapper_x, wrapper_x + wrapper_w, wrapper_y + wrapper_h + 60, "1538 px")
    if notes:
        text(draw, 82, y + 158, notes, font=load_font("bold", 14), fill=PALETTE["muted"], anchor="lm")


def draw_shell(active_nav: str, title: str, subtitle: str, size=(1920, 1080)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", size, PALETTE["bg_top"])
    gradient_background(img, PALETTE["bg_top"], PALETTE["bg_bottom"])
    draw = ImageDraw.Draw(img)
    w, h = size

    # Top bar
    rr(draw, (24, 20, w - 24, 96), fill=(250, 252, 255), outline=(215, 227, 236), width=1, radius=22)
    text(draw, 56, 54, "Tile Compile GUI 2", font=load_font("serif", 34), fill=PALETTE["ink"], anchor="lm")
    chip(draw, 360, 37, "Modernized Workspace", kind="primary")
    chip(draw, 1460, 37, "DE", kind="primary")
    chip(draw, 1532, 37, "EN", kind="neutral")
    chip(draw, 1620, 37, "Run Ready", kind="ok")

    # Sidebar
    rr(draw, (24, 116, 324, h - 24), fill=(247, 251, 255), outline=(212, 225, 236), width=1, radius=20)
    text(draw, 52, 154, "Navigation", font=load_font("bold", 19), fill=PALETTE["muted"], anchor="lm")
    nav_items = [
        "Dashboard",
        "Input & Scan",
        "Parameter Studio",
        "Assumptions",
        "Run Monitor",
        "History",
        "Astrometry & PCC",
        "Live Log",
    ]
    y = 194
    for item in nav_items:
        is_active = item == active_nav
        fill = PALETTE["primary_soft"] if is_active else (247, 251, 255)
        outline = (173, 214, 220) if is_active else (247, 251, 255)
        rr(draw, (44, y - 20, 304, y + 22), fill=fill, outline=outline, width=1, radius=12)
        icon_col = PALETTE["primary"] if is_active else (141, 153, 168)
        draw.ellipse((56, y - 7, 70, y + 7), fill=icon_col)
        text(draw, 82, y + 2, item, font=load_font("bold", 18), fill=PALETTE["ink"], anchor="lm")
        y += 56

    rr(draw, (44, h - 182, 304, h - 52), fill=(255, 255, 255), outline=PALETTE["line"], radius=12)
    text(draw, 58, h - 156, "Quick Action", font=load_font("bold", 17), fill=PALETTE["muted"], anchor="lm")
    text(draw, 58, h - 126, "New Guided Run", font=load_font("bold", 20), fill=PALETTE["ink"], anchor="lm")
    button(draw, 58, h - 110, 152, 44, "Start Wizard", primary=True)

    # Header in main area
    text(draw, 358, 145, title, font=load_font("serif", 42), fill=PALETTE["ink"], anchor="lm")
    text(draw, 358, 182, subtitle, font=load_font("regular", 22), fill=PALETTE["muted"], anchor="lm")

    return img, draw


def mockup_styleboard():
    img = Image.new("RGB", (1920, 1080), PALETTE["bg_top"])
    gradient_background(img, (244, 247, 252), (229, 237, 244))
    draw = ImageDraw.Draw(img)

    text(draw, 72, 70, "GUI 2 Linie B", font=load_font("serif", 52), fill=PALETTE["ink"], anchor="lm")
    text(
        draw,
        72,
        122,
        "Finale Designlinie: Parameter Studio mit Explain-Panel, Situation Assistant und i18n.",
        font=load_font("regular", 23),
        fill=PALETTE["muted"],
        anchor="lm",
    )

    # Main B card.
    rr(draw, (72, 180, 1240, 920), fill=(255, 255, 255), outline=PALETTE["line"], radius=20)
    rr(draw, (96, 214, 1212, 456), fill=(242, 249, 252), outline=(220, 229, 238), radius=14)
    rr(draw, (120, 242, 500, 286), fill=(255, 255, 255), outline=(220, 229, 238), radius=10)
    rr(draw, (520, 242, 860, 286), fill=(255, 255, 255), outline=(220, 229, 238), radius=10)
    rr(draw, (880, 242, 1180, 286), fill=PALETTE["primary"], outline=None, radius=10)
    rr(draw, (120, 310, 1180, 430), fill=(255, 255, 255), outline=(220, 229, 238), radius=10)

    text(draw, 108, 508, "Linie B - Parameter Studio", font=load_font("serif", 42), fill=PALETTE["ink"], anchor="lm")
    draw_paragraph(
        draw,
        108,
        566,
        "Zentraler Arbeitsraum fuer alle Parameter mit Suchleiste, Presets, Guardrails, Kurz-Erklaerungen, Situation Assistant und i18n.",
        font=load_font("regular", 25),
        fill=PALETTE["muted"],
        max_width=1080,
        line_height=36,
    )
    for i, l in enumerate([
        "+ Vollstaendige Parametereingabe",
        "+ Kontext-Hilfe pro Feld",
        "+ Szenario-Empfehlungen (Alt/Az, Rotation, helle Sterne)",
        "+ Live i18n (DE/EN)",
    ]):
        text(draw, 112, 680 + i * 46, l, font=load_font("bold", 24), fill=(67, 92, 112), anchor="lm")
    chip(draw, 110, 860, "Klickdummy abgestimmt", kind="primary")

    # Right side summary.
    rr(draw, (1270, 180, 1848, 920), fill=(255, 255, 255), outline=PALETTE["line"], radius=20)
    text(draw, 1298, 234, "Abgestimmte Screens", font=load_font("serif", 33), fill=PALETTE["ink"], anchor="lm")
    cards = [
        "Dashboard",
        "Parameter Studio",
        "Run Monitor",
        "History + Tools",
        "Flow Overview",
        "Layout 1920 Spec",
    ]
    yy = 286
    for c in cards:
        rr(draw, (1298, yy, 1820, yy + 82), fill=(247, 252, 255), outline=(220, 229, 238), radius=12)
        text(draw, 1322, yy + 43, c, font=load_font("bold", 23), fill=PALETTE["ink"], anchor="lm")
        chip(draw, 1690, yy + 25, "B", kind="ok")
        yy += 96

    img.save(OUT / "gui2_01_styleboard.png")


def mockup_dashboard():
    img, draw = draw_shell(
        "Dashboard",
        "Projekt Dashboard",
        "Schnellstart, Run-Readiness und Status aller Kernmodule in einem Blick.",
    )

    # Main content cards.
    metric_card(draw, 356, 230, 286, 150, "Frames entdeckt", "1,264", "+3.2%", trend="ok")
    metric_card(draw, 662, 230, 286, 150, "Scan-Qualitat", "0.91", "stabil", trend="primary")
    metric_card(draw, 968, 230, 286, 150, "Offene Warnungen", "4", "check", trend="warn")
    metric_card(draw, 1274, 230, 286, 150, "Letzter Lauf", "OK", "2h 13m", trend="ok")

    rr(draw, (356, 402, 1210, 860), fill=PALETTE["surface"], outline=PALETTE["line"], radius=18)
    text(draw, 384, 438, "Guided Run", font=load_font("serif", 34), fill=PALETTE["ink"], anchor="lm")
    text(
        draw,
        384,
        472,
        "1) Input/Run-Ziel  2) Preset + Parameter  3) Validieren  4) Batch starten",
        font=load_font("regular", 20),
        fill=PALETTE["muted"],
        anchor="lm",
    )

    input_field(draw, 384, 528, 560, 40, "Input Dirs", "/data/m31/2026-03-01, /data/m31/2026-03-02")
    input_field(draw, 384, 620, 560, 40, "Runs Dir", "/data/tile_runs")
    input_field(draw, 384, 712, 172, 40, "Farbmodus", "MONO")
    input_field(draw, 568, 712, 188, 40, "Run Name", "M31_altaz_test")
    input_field(draw, 768, 712, 176, 40, "Preset", "Smart Scope")
    rr(draw, (384, 764, 734, 786), fill=(247, 252, 255), outline=(220, 229, 238), radius=8)
    text(draw, 396, 775, "Queue: L -> R -> G -> B -> Ha", font=load_font("bold", 14), fill=PALETTE["muted"], anchor="lm")
    rr(draw, (744, 764, 944, 786), fill=(247, 252, 255), outline=(220, 229, 238), radius=8)
    text(draw, 756, 775, "M31_altaz_test_20260307_221530", font=load_font("bold", 10), fill=PALETTE["muted"], anchor="lm")
    button(draw, 384, 796, 182, 52, "Scan neu")
    button(draw, 578, 796, 214, 52, "Parameter Studio")
    button(draw, 804, 796, 194, 52, "Run starten", primary=True)

    rr(draw, (1230, 402, 1894, 860), fill=PALETTE["surface"], outline=PALETTE["line"], radius=18)
    text(draw, 1258, 438, "Readiness Guardrails", font=load_font("serif", 33), fill=PALETTE["ink"], anchor="lm")

    y = SPACING_1920["dashboard"]["readiness_first_row_y"]
    readiness_row_h = SPACING_1920["dashboard"]["readiness_row_h"]
    readiness_row_step = SPACING_1920["dashboard"]["readiness_row_step"]
    checks = [
        ("Scan erfolgreich", "ok"),
        ("Color mode bestaetigt", "ok"),
        ("Config validiert", "warn"),
        ("Kalibrierpfade vollstandig", "ok"),
        ("BGE/PCC Werte im Rahmen", "warn"),
    ]
    for label, kind in checks:
        rr(draw, (1258, y, 1868, y + readiness_row_h), fill=(252, 254, 255), outline=(223, 232, 239), radius=12)
        chip(draw, 1274, y + 6, "OK" if kind == "ok" else "CHECK", kind=kind)
        text(draw, 1394, y + 24, label, font=load_font("bold", 19), fill=PALETTE["ink"], anchor="lm")
        y += readiness_row_step

    rr(draw, (356, 880, 1894, 1036), fill=PALETTE["surface"], outline=PALETTE["line"], radius=18)
    text(draw, 384, 916, "Pipeline Vorschau", font=load_font("serif", 32), fill=PALETTE["ink"], anchor="lm")
    phases = ["SCAN", "REG", "TILES", "STACK", "ASTROM", "BGE", "PCC", "DONE"]
    x = 384
    for i, ph in enumerate(phases):
        kind = "primary" if i <= 2 else "neutral"
        chip(draw, x, 960, ph, kind=kind)
        x += 172
        if i < len(phases) - 1:
            draw.line((x - 18, 977, x - 6, 977), fill=(156, 177, 192), width=3)

    img.save(OUT / "gui2_02_dashboard.png")


def mockup_parameter_studio():
    img, draw = draw_shell(
        "Parameter Studio",
        "Parameter Studio",
        "Alle verwendeten Parameter in einer konsistenten, filterbaren und validierbaren Oberflache.",
    )

    rr(draw, (356, 230, 1894, 1028), fill=PALETTE["surface"], outline=PALETTE["line"], radius=18)

    # Top toolbar
    input_field(draw, 382, 272, 490, 54, "Suche", "z.B. pcc.sigma_clip, bge.fit.rbf_lambda")
    button(draw, 886, 272, 126, 54, "Preset")
    button(draw, 1026, 272, 154, 54, "Situation")
    button(draw, 1194, 272, 172, 54, "YAML Sync")
    button(draw, 1380, 272, 164, 54, "Validieren", primary=True)
    chip(draw, 1558, 282, "127 Parameter", kind="primary")
    chip(draw, 1722, 282, "3 Warnungen", kind="warn")

    # Left category rail
    rr(draw, (382, 346, 364 + 350, 996), fill=PALETTE["surface_alt"], outline=(223, 232, 239), radius=12)
    cats = [
        "Pipeline",
        "Input & Calibration",
        "Assumptions",
        "Registration",
        "Tile & Metrics",
        "Synthetic",
        "Astrometry",
        "BGE",
        "PCC",
        "Stacking",
        "Runtime Limits",
    ]
    y = 374
    for c in cats:
        active = c in {"Registration", "BGE", "PCC"}
        fill = PALETTE["primary_soft"] if active else PALETTE["surface_alt"]
        rr(draw, (396, y, 700, y + 40), fill=fill, outline=None, radius=10)
        text(draw, 416, y + 21, c, font=load_font("bold", 18), fill=PALETTE["ink"], anchor="lm")
        y += 52

    # Main form area
    rr(draw, (748, 346, 1380, 996), fill=(255, 255, 255), outline=(223, 232, 239), radius=12)
    text(draw, 772, 380, "Registration", font=load_font("serif", 30), fill=PALETTE["ink"], anchor="lm")
    reg_row1_y = SPACING_1920["parameter_studio"]["registration_row1_y"]
    reg_row2_y = SPACING_1920["parameter_studio"]["registration_row2_y"]
    reg_row2_gap = SPACING_1920["parameter_studio"]["registration_row2_hgap"]
    section_gap = SPACING_1920["parameter_studio"]["section_title_gap"]

    input_field(draw, 772, reg_row1_y, 280, 52, "engine", "robust_phase_ecc")
    input_field(draw, 1068, reg_row1_y, 280, 52, "allow_rotation", "true")
    # Registration row 2 intentionally uses larger vertical/horizontal gaps for readability.
    reg_x0 = 772
    reg_w1 = 184
    reg_w2 = 184
    reg_w3 = 176
    reg_x1 = reg_x0 + reg_w1 + reg_row2_gap
    reg_x2 = reg_x1 + reg_w2 + reg_row2_gap
    input_field(draw, reg_x0, reg_row2_y, reg_w1, 52, "star_topk", "180")
    input_field(draw, reg_x1, reg_row2_y, reg_w2, 52, "star_inlier_tol_px", "4.0")
    input_field(draw, reg_x2, reg_row2_y, reg_w3, 52, "reject_cc_min_abs", "0.30")

    bge_title_y = reg_row2_y + 52 + section_gap
    bge_row_y = bge_title_y + 42
    text(draw, 772, bge_title_y, "BGE", font=load_font("serif", 30), fill=PALETTE["ink"], anchor="lm")
    input_field(draw, 772, bge_row_y, 188, 52, "enabled", "true")
    input_field(draw, 974, bge_row_y, 188, 52, "fit.method", "rbf")
    input_field(draw, 1176, bge_row_y, 172, 52, "rbf_lambda", "1e-2")

    pcc_title_y = bge_row_y + 52 + section_gap
    pcc_row_y = pcc_title_y + 42
    text(draw, 772, pcc_title_y, "PCC", font=load_font("serif", 30), fill=PALETTE["ink"], anchor="lm")
    input_field(draw, 772, pcc_row_y, 188, 52, "source", "siril")
    input_field(draw, 974, pcc_row_y, 188, 52, "sigma_clip", "2.5")
    input_field(draw, 1176, pcc_row_y, 172, 52, "k_max", "2.4")

    action_row_y = pcc_row_y + 52 + 84
    # Keep total width stable but allocate more room for the long middle label.
    button(draw, 772, action_row_y, 170, 54, "Auf Default")
    button(draw, 954, action_row_y, 214, 54, "Aenderungen pruefen")
    button(draw, 1180, action_row_y, 170, 54, "Speichern", primary=True)

    # Right explain + situation panel
    rr(draw, (1398, 346, 1868, 996), fill=PALETTE["surface_alt"], outline=(223, 232, 239), radius=12)
    text(draw, 1422, 380, "Explain, Situation & i18n", font=load_font("serif", 26), fill=PALETTE["ink"], anchor="lm")
    rr(draw, (1422, 402, 1848, 558), fill=(255, 255, 255), outline=(212, 225, 236), radius=10)
    text(draw, 1438, 428, "Parameter", font=load_font("bold", 18), fill=PALETTE["muted"], anchor="lm")
    text(draw, 1536, 428, "registration.allow_rotation", font=load_font("bold", 17), fill=PALETTE["ink"], anchor="lm")
    draw_paragraph(
        draw,
        1438,
        456,
        "Kurz: Erlaubt Rotationskomponente im Warp-Modell. Wichtig fuer Alt/Az oder starke Feldrotation.",
        font=load_font("regular", 17),
        fill=PALETTE["muted"],
        max_width=390,
        line_height=24,
    )
    chip(draw, 1438, 518, "Range OK", kind="ok")
    chip(draw, 1556, 518, "Spec v3.3.6", kind="primary")
    chip(draw, 1706, 518, "Alt/Az: true", kind="warn")

    rr(draw, (1422, 576, 1848, 784), fill=(255, 255, 255), outline=(212, 225, 236), radius=10)
    text(draw, 1438, 602, "Situation Assistant", font=load_font("bold", 22), fill=PALETTE["ink"], anchor="lm")
    chip(draw, 1438, 618, "Alt/Az", kind="primary")
    chip(draw, 1538, 618, "Starke Rotation", kind="primary")
    chip(draw, 1708, 618, "Helle Sterne", kind="warn")
    draw_paragraph(
        draw,
        1438,
        666,
        "Empfohlen: registration.allow_rotation=true, star_topk=180, reject_shift_px_min=120.",
        font=load_font("regular", 16),
        fill=PALETTE["muted"],
        max_width=390,
        line_height=24,
    )
    chip(draw, 1438, 742, "DE", kind="primary")
    chip(draw, 1510, 742, "EN", kind="neutral")

    text(draw, 1422, 818, "YAML Diff", font=load_font("bold", 22), fill=PALETTE["ink"], anchor="lm")
    rr(draw, (1422, 838, 1848, 972), fill=(255, 255, 255), outline=(212, 225, 236), radius=10)
    mono = load_font("regular", 17)
    diff_lines = [
        "- registration.engine: triangle_star_matching",
        "+ registration.engine: robust_phase_ecc",
        "+ registration.allow_rotation: true",
        "+ registration.star_topk: 180",
    ]
    yy = 866
    for ln in diff_lines:
        color = PALETTE["error"] if ln.startswith("-") else PALETTE["ok"]
        text(draw, 1438, yy, ln, font=mono, fill=color, anchor="lm")
        yy += 30

    img.save(OUT / "gui2_03_parameter_studio.png")


def mockup_run_monitor():
    img, draw = draw_shell(
        "Run Monitor",
        "Run Monitor",
        "Live-Phasenansicht mit Batch-Kontext, Logs, Stats und Resume-Einstiegspunkten.",
    )

    rr(draw, (356, 230, 1894, 1028), fill=PALETTE["surface"], outline=PALETTE["line"], radius=18)

    # Batch strip
    rr(draw, (382, 266, 1868, 338), fill=(247, 252, 255), outline=(217, 229, 238), radius=12)
    text(draw, 404, 296, "Batch 2/5  |  Filter 2/5: R  |  input: /data/m31/R  |  run_id: 20260307_ab12cd34/m31_R", font=load_font("bold", 20), fill=PALETTE["ink"], anchor="lm")
    chip(draw, 1580, 280, "running", kind="primary")
    button(draw, 1700, 276, 144, 48, "Stop")

    # Left phases
    rr(draw, (382, 358, 1030, 996), fill=(255, 255, 255), outline=(223, 232, 239), radius=12)
    text(draw, 406, 392, "Phasen", font=load_font("serif", 31), fill=PALETTE["ink"], anchor="lm")
    phases = [
        ("SCAN_INPUT", "ok"),
        ("CHANNEL_SPLIT", "ok"),
        ("NORMALIZATION", "ok"),
        ("GLOBAL_METRICS", "running"),
        ("TILE_GRID", "pending"),
        ("REGISTRATION", "pending"),
        ("STACKING", "pending"),
        ("BGE", "pending"),
        ("PCC", "pending"),
    ]
    y = 432
    for ph, st in phases:
        rr(draw, (406, y, 1000, y + 52), fill=(251, 253, 255), outline=(226, 235, 242), radius=10)
        text(draw, 426, y + 27, ph, font=load_font("bold", 18), fill=PALETTE["ink"], anchor="lm")
        chip(draw, 836, y + 9, st.upper(), kind=("ok" if st == "ok" else "primary" if st == "running" else "neutral"))
        if st == "running":
            rr(draw, (640, y + 18, 812, y + 34), fill=(229, 236, 243), radius=8)
            rr(draw, (640, y + 18, 734, y + 34), fill=PALETTE["primary"], radius=8)
        y += 60

    # Center column: live log (top) + stats (below log).
    rr(draw, (1048, 358, 1468, 996), fill=PALETTE["surface_alt"], outline=(223, 232, 239), radius=12)
    text(draw, 1072, 392, "Live Log", font=load_font("serif", 31), fill=PALETTE["ink"], anchor="lm")
    rr(draw, (1072, 424, 1444, 628), fill=(255, 255, 255), outline=(215, 227, 236), radius=10)
    mono = load_font("regular", 16)
    lines = [
        "[09:21:34] run_start status=running",
        "[09:21:45] phase_start GLOBAL_METRICS",
        "[09:22:10] phase_progress 37/120",
        "[09:22:38] phase_progress 64/120",
        "[09:23:18] warning: high background variance",
    ]
    yy = 452
    for ln in lines:
        col = PALETTE["warn"] if "warning" in ln else (72, 86, 103)
        text(draw, 1090, yy, ln, font=mono, fill=col, anchor="lm")
        yy += 34

    rr(draw, (1072, 646, 1444, 968), fill=(255, 255, 255), outline=(215, 227, 236), radius=10)
    text(draw, 1090, 678, "Stats", font=load_font("serif", 28), fill=PALETTE["ink"], anchor="lm")
    rr(draw, (1084, 716, 1434, 772), fill=(255, 255, 255), outline=(215, 227, 236), radius=8)
    text(draw, 1096, 744, "Aktiver Run: 20260307_ab12", font=load_font("bold", 15), fill=PALETTE["ink"], anchor="lm")
    rr(draw, (1084, 788, 1434, 844), fill=(255, 255, 255), outline=(215, 227, 236), radius=8)
    text(draw, 1096, 816, "Stats Script: python3 .../scripts/generate_report.py", font=load_font("bold", 12), fill=PALETTE["muted"], anchor="lm")
    button(draw, 1084, 860, 166, 52, "Generate Stats")
    button(draw, 1260, 860, 174, 52, "Open Stats Folder")

    # Right artifacts and controls
    rr(draw, (1486, 358, 1868, 996), fill=PALETTE["surface_alt"], outline=(223, 232, 239), radius=12)
    text(draw, 1510, 392, "Artefakte", font=load_font("serif", 30), fill=PALETTE["ink"], anchor="lm")
    rr(draw, (1510, 422, 1842, 470), fill=(247, 252, 255), outline=(215, 227, 236), radius=10)
    text(draw, 1528, 446, "Queue: L=OK, R=RUNNING, G/B/Ha=PENDING", font=load_font("bold", 13), fill=PALETTE["muted"], anchor="lm")
    rr(draw, (1510, 480, 1842, 528), fill=(255, 255, 255), outline=(215, 227, 236), radius=10)
    text(draw, 1528, 504, "Config-Revision: rev_20260307_0912 (select)", font=load_font("bold", 14), fill=PALETTE["ink"], anchor="lm")
    cards = [
        "config_revisions.json  11 KB",
        "reconstructed_R.fit  98 MB",
        "reconstructed_G.fit  97 MB",
        "reconstructed_B.fit  96 MB",
    ]
    y = 538
    for c in cards:
        rr(draw, (1510, y, 1842, y + 52), fill=(255, 255, 255), outline=(215, 227, 236), radius=10)
        text(draw, 1528, y + 27, c, font=load_font("bold", 16), fill=PALETTE["ink"], anchor="lm")
        y += 62
    action_y = SPACING_1920["run_monitor"]["artifact_button_row_y"]
    action_y2 = SPACING_1920["run_monitor"]["artifact_secondary_button_y"]
    button(draw, 1510, action_y, 158, 50, "Resume")
    button(draw, 1682, action_y, 160, 50, "Report", primary=True)
    button(draw, 1510, action_y2, 160, 50, "Restore Rev")
    button(draw, 1682, action_y2, 160, 50, "Run-Ordner", primary=False)

    img.save(OUT / "gui2_04_run_monitor.png")


def mockup_history_tools():
    img, draw = draw_shell(
        "History",
        "Run History + Tools",
        "Historie, Detailanalyse und Astrometry/PCC-Tools in einem modernisierten Workspace.",
    )

    rr(draw, (356, 230, 1894, 1028), fill=PALETTE["surface"], outline=PALETTE["line"], radius=18)

    rr(draw, (382, 266, 1242, 996), fill=(255, 255, 255), outline=(223, 232, 239), radius=12)
    text(draw, 406, 300, "Run Historie", font=load_font("serif", 32), fill=PALETTE["ink"], anchor="lm")
    header_y = 340
    rr(draw, (406, header_y, 1218, header_y + 42), fill=(247, 252, 255), outline=(223, 232, 239), radius=8)
    cols = ["run_id", "timestamp", "status", "frames", "phases"]
    x = [424, 640, 826, 958, 1062]
    for i, c in enumerate(cols):
        text(draw, x[i], header_y + 21, c, font=load_font("bold", 17), fill=PALETTE["muted"], anchor="lm")

    rows = [
        ("20260307_ab12", "09:21", "running", "1264", "4/18"),
        ("20260306_f93e", "22:47", "completed", "1188", "18/18"),
        ("20260305_a124", "20:10", "error", "902", "11/18"),
        ("20260304_991f", "18:04", "completed", "1310", "18/18"),
        ("20260303_aa71", "14:55", "completed", "1004", "18/18"),
    ]
    y = 392
    for r in rows:
        rr(draw, (406, y, 1218, y + 52), fill=(251, 253, 255), outline=(228, 236, 242), radius=8)
        for i, v in enumerate(r):
            if i == 2:
                chip(draw, 808, y + 9, v.upper(), kind=("ok" if v == "completed" else "error" if v == "error" else "primary"))
            else:
                text(draw, x[i], y + 27, v, font=load_font("bold", 17), fill=PALETTE["ink"], anchor="lm")
        y += 60

    button(draw, 406, 930, 164, 50, "Refresh")
    button(draw, 584, 930, 210, 50, "Als Current Run")
    button(draw, 808, 930, 210, 50, "Report offnen", primary=True)

    rr(draw, (1260, 266, 1868, 648), fill=PALETTE["surface_alt"], outline=(223, 232, 239), radius=12)
    text(draw, 1284, 300, "Astrometry", font=load_font("serif", 31), fill=PALETTE["ink"], anchor="lm")
    ast_y1 = SPACING_1920["history_tools"]["astrometry_first_input_y"]
    ast_step = SPACING_1920["history_tools"]["astrometry_row_step"]
    plate_y = SPACING_1920["history_tools"]["astrometry_plate_solve_y"]
    input_field(draw, 1284, ast_y1, 560, 52, "ASTAP binary", "~/.local/share/tile_compile/astap/astap_cli")
    input_field(draw, 1284, ast_y1 + ast_step, 560, 52, "Catalog", "D50 installed")
    input_field(draw, 1284, plate_y, 368, 52, "Plate Solve File", "stacked_m31.fits")
    button(draw, 1664, plate_y, 180, 52, "Solve", primary=True)

    rr(draw, (1260, 666, 1868, 996), fill=PALETTE["surface_alt"], outline=(223, 232, 239), radius=12)
    text(draw, 1284, 700, "PCC", font=load_font("serif", 31), fill=PALETTE["ink"], anchor="lm")
    input_field(draw, 1284, 744, 180, 52, "source", "siril")
    input_field(draw, 1476, 744, 180, 52, "sigma", "2.5")
    input_field(draw, 1668, 744, 176, 52, "min_stars", "10")
    button(draw, 1284, 826, 260, 52, "Run PCC", primary=True)
    button(draw, 1558, 826, 286, 52, "Save Corrected")
    rr(draw, (1284, 890, 560 + 1284, 930), fill=(255, 255, 255), outline=(215, 227, 236), radius=8)
    text(draw, 1300, 910, "Stats sind im Run Monitor unter Live Log verfuegbar.", font=load_font("bold", 14), fill=PALETTE["muted"], anchor="lm")

    img.save(OUT / "gui2_05_history_tools.png")


def mockup_flow_overview():
    img = Image.new("RGB", (1920, 1080), PALETTE["bg_top"])
    gradient_background(img, (244, 248, 252), (230, 238, 246))
    draw = ImageDraw.Draw(img)

    text(draw, 64, 68, "Ablauf-Blueprint GUI 2", font=load_font("serif", 50), fill=PALETTE["ink"], anchor="lm")
    text(draw, 64, 122, "Vom Input bis zu Report und Wiederaufnahme, inklusive MONO-Filter-Queue, Guardrails und Parameter-Studio.", font=load_font("regular", 24), fill=PALETTE["muted"], anchor="lm")

    nodes = [
        (80, 220, 300, 132, "Input & Scan", "Ordner, Pattern, run_name"),
        (428, 220, 300, 132, "Calibration", "Bias/Dark/Flat Pfade"),
        (776, 220, 300, 132, "Parameter Studio", "Alle Config Keys + Queue"),
        (1124, 220, 300, 132, "Validation", "Schema + Guardrails"),
        (1472, 220, 300, 132, "Run Start", "runs_dir + Name+Datum"),
        (1472, 438, 300, 132, "Run Monitor", "Phasen, Live Log, Stats"),
        (1124, 438, 300, 132, "History", "Vergleich, Selektion"),
        (776, 438, 300, 132, "Current Run", "Resume (Filter+Phase+Revision)"),
        (428, 438, 300, 132, "Astrometry/PCC", "Tools + Quicktests"),
        (80, 438, 300, 132, "Export", "Config, Diff, Reports"),
    ]

    for x, y, w, h, title, sub in nodes:
        rr(draw, (x, y, x + w, y + h), fill=PALETTE["surface"], outline=PALETTE["line"], radius=16)
        text(draw, x + 20, y + 44, title, font=load_font("serif", 30), fill=PALETTE["ink"], anchor="lm")
        text(draw, x + 20, y + 84, sub, font=load_font("regular", 19), fill=PALETTE["muted"], anchor="lm")

    # Arrows
    arrows = [
        ((380, 286), (426, 286)),
        ((728, 286), (774, 286)),
        ((1076, 286), (1122, 286)),
        ((1424, 286), (1470, 286)),
        ((1622, 354), (1622, 436)),
        ((1470, 504), (1426, 504)),
        ((1122, 504), (1078, 504)),
        ((774, 504), (730, 504)),
        ((426, 504), (382, 504)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        draw.line((x1, y1, x2, y2), fill=(120, 145, 168), width=6)
        # simple arrow head
        if x2 > x1:
            draw.polygon([(x2, y2), (x2 - 14, y2 - 8), (x2 - 14, y2 + 8)], fill=(120, 145, 168))
        elif x2 < x1:
            draw.polygon([(x2, y2), (x2 + 14, y2 - 8), (x2 + 14, y2 + 8)], fill=(120, 145, 168))
        elif y2 > y1:
            draw.polygon([(x2, y2), (x2 - 8, y2 - 14), (x2 + 8, y2 - 14)], fill=(120, 145, 168))
        else:
            draw.polygon([(x2, y2), (x2 - 8, y2 + 14), (x2 + 8, y2 + 14)], fill=(120, 145, 168))

    rr(draw, (64, 648, 1856, 994), fill=PALETTE["surface"], outline=PALETTE["line"], radius=16)
    text(draw, 92, 690, "Kernprinzipien", font=load_font("serif", 38), fill=PALETTE["ink"], anchor="lm")
    principles = [
        "Ein Parameter, ein eindeutiger Eingabepunkt: keine versteckten YAML-only Optionen ohne Form-Alternative.",
        "Gefuhrte Standardpfade + Expert Mode: Einsteiger schnell, Expert:innen vollstandig.",
        "Kontinuierliche Validierung mit unmittelbarer Rueckmeldung vor dem Run-Start.",
        "MONO-Filter werden als Queue strikt seriell abgearbeitet (Filter i/N) mit Resume je Filter+Phase.",
        "Batch- und Einzelrun teilen dieselbe Oberflache, nur Kontextpanel aendert sich.",
    ]
    yy = 744
    for p in principles:
        text(draw, 100, yy, "-", font=load_font("serif", 34), fill=PALETTE["primary"], anchor="lm")
        draw_paragraph(
            draw,
            128,
            yy,
            p,
            font=load_font("regular", 24),
            fill=PALETTE["muted"],
            max_width=1690,
            line_height=34,
        )
        yy += 66

    img.save(OUT / "gui2_07_flow_overview.png")


def mockup_layout_1920_overlay():
    img, draw = draw_shell(
        "Dashboard",
        "Layout 1920 Spec Overlay",
        "Fixes Desktop-Raster (1920x1080) mit Pixelkoordinaten, Spaltenbreiten und Interaktionszonen.",
    )

    sidebar = LAYOUT_1920["sidebar"]
    main_area = LAYOUT_1920["main_area"]
    wrapper = LAYOUT_1920["main_wrapper"]
    dash_guided = LAYOUT_1920["dashboard_guided"]
    dash_guard = LAYOUT_1920["dashboard_guard"]

    # Spec overlay in shell.
    rr(draw, sidebar, fill=(232, 244, 249), outline=(149, 198, 208), width=2, radius=18)
    rr(draw, main_area, fill=(244, 250, 255), outline=(157, 184, 208), width=2, radius=18)
    rr(draw, wrapper, fill=(255, 255, 255), outline=(117, 157, 189), width=3, radius=18)
    rr(draw, dash_guided, fill=(236, 247, 255), outline=(136, 180, 206), width=2, radius=14)
    rr(draw, dash_guard, fill=(250, 243, 235), outline=(210, 170, 136), width=2, radius=14)

    chip(draw, 46, 122, "Sidebar 300 px", kind="primary")
    chip(draw, 362, 122, "Main Area 1538 px", kind="primary")
    chip(draw, 362, 236, "Main Wrapper 1538 x 798", kind="warn")
    text(draw, 50, 176, "x=24..324 | y=116..1056", font=load_font("bold", 16), fill=PALETTE["muted"], anchor="lm")
    text(draw, 362, 176, "x=356..1894 | y=116..1056", font=load_font("bold", 16), fill=PALETTE["muted"], anchor="lm")
    text(draw, 362, 264, "Wrapper Start: x=356, y=230", font=load_font("bold", 16), fill=PALETTE["muted"], anchor="lm")

    # Detailed mapping panel.
    rr(draw, (356, 690, 1894, 1028), fill=PALETTE["surface"], outline=PALETTE["line"], radius=16)
    text(draw, 384, 724, "Screen Panel Mapping (1920)", font=load_font("serif", 32), fill=PALETTE["ink"], anchor="lm")

    split_row(
        draw,
        384,
        754,
        1120,
        38,
        "Dashboard:",
        [
            (854, "Guided 854", (232, 244, 249)),
            (664, "Guardrails 664", (250, 243, 235)),
        ],
        gaps=[20],
    )
    split_row(
        draw,
        384,
        806,
        1120,
        38,
        "Parameter:",
        [
            (350, "Kategorien 350", (237, 244, 255)),
            (632, "Form 632", (231, 248, 238)),
            (470, "Explain 470", (251, 243, 236)),
        ],
        gaps=[18, 18],
    )
    split_row(
        draw,
        384,
        858,
        1120,
        38,
        "Run Monitor:",
        [
            (648, "Phasen 648", (237, 244, 255)),
            (420, "Live+Stats 420", (231, 248, 238)),
            (382, "Artefakte 382", (251, 243, 236)),
        ],
        gaps=[18, 18],
    )
    split_row(
        draw,
        384,
        910,
        1120,
        38,
        "History+Tools:",
        [
            (860, "History 860", (237, 244, 255)),
            (608, "Tools 608", (251, 243, 236)),
        ],
        gaps=[18],
    )

    rr(draw, (384, 960, 1844, 1006), fill=(247, 252, 255), outline=(217, 229, 238), radius=10)
    text(
        draw,
        400,
        983,
        "Interaktionszonen: Primary >=52 px, Secondary >=44 px, Zeilenaktionen >=48 px, Chips >=34 px.",
        font=load_font("bold", 15),
        fill=PALETTE["muted"],
        anchor="lm",
    )

    # Topbar and shell dimensions.
    draw.line((24, 106, 1896, 106), fill=(145, 165, 182), width=2)
    text(draw, 34, 106, "Workspace y=116", font=load_font("bold", 14), fill=PALETTE["muted"], anchor="ls")
    draw.line((24, 99, 24, 20), fill=(145, 165, 182), width=2)
    text(draw, 36, 62, "Topbar 76 px", font=load_font("bold", 14), fill=PALETTE["muted"], anchor="lm")

    img.save(OUT / "gui2_08_layout_1920_overlay.png")


def mockup_layout_1920_measurelines():
    img = Image.new("RGB", (1920, 1080), PALETTE["bg_top"])
    gradient_background(img, (244, 248, 252), (230, 238, 246))
    draw = ImageDraw.Draw(img)

    text(draw, 64, 66, "Layout 1920 Measurelines", font=load_font("serif", 50), fill=PALETTE["ink"], anchor="lm")
    text(
        draw,
        64,
        120,
        "Explizite Pixelmasslinien je Screen fuer Panelbreiten, Gutter und Inset-Werte (Desktop 1920x1080).",
        font=load_font("regular", 22),
        fill=PALETTE["muted"],
        anchor="lm",
    )

    draw_measure_row(
        draw,
        156,
        "Dashboard",
        [
            ("Guided", 854, (231, 244, 255)),
            ("Guardrails", 664, (251, 243, 235)),
        ],
        start_offset=0,
        gaps=[20],
        notes="Panelhoehe real: 382 px | Gap: 20 px",
    )
    draw_measure_row(
        draw,
        356,
        "Parameter Studio",
        [
            ("Kategorien", 350, (236, 244, 255)),
            ("Form", 632, (232, 248, 238)),
            ("Explain", 470, (251, 243, 236)),
        ],
        start_offset=26,
        gaps=[16, 18],
        notes="Panelhoehen real: 650 px | Row-2 Registration Gap: 20 px",
    )
    draw_measure_row(
        draw,
        556,
        "Run Monitor",
        [
            ("Phasen", 648, (236, 244, 255)),
            ("Live+Stats", 420, (232, 248, 238)),
            ("Artefakte", 382, (251, 243, 236)),
        ],
        start_offset=26,
        gaps=[18, 18],
        notes="Panelhoehen real: 638 px | Mittelspalte: Live Log oben + Stats unten",
    )
    draw_measure_row(
        draw,
        756,
        "History + Tools",
        [
            ("History", 860, (236, 244, 255)),
            ("Tools", 608, (251, 243, 236)),
        ],
        start_offset=26,
        gaps=[18],
        notes="Panelhoehen real: History 730 px | Tools oben 382 px | Tools unten 330 px | V-Gap 18 px",
    )

    rr(draw, (64, 966, 1856, 1030), fill=PALETTE["surface"], outline=PALETTE["line"], radius=12)
    text(
        draw,
        92,
        998,
        "Interaktionszonen: Primary >=52 px, Secondary >=44 px, Row Actions >=48 px, Chips >=34 px.",
        font=load_font("bold", 16),
        fill=PALETTE["muted"],
        anchor="lm",
    )

    img.save(OUT / "gui2_09_layout_1920_measurelines.png")


def main():
    mockup_styleboard()
    mockup_dashboard()
    mockup_parameter_studio()
    mockup_run_monitor()
    mockup_history_tools()
    mockup_flow_overview()
    mockup_layout_1920_overlay()
    mockup_layout_1920_measurelines()
    print(f"Generated mockups in {OUT}")


if __name__ == "__main__":
    main()
