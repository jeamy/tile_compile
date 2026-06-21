#!/usr/bin/env python3
"""Generate GUI3 mockup PNGs using matplotlib."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

# ─── Design Tokens ───────────────────────────────────────────────
OUT = os.path.dirname(os.path.abspath(__file__))

THEMES = {
    "light": {
        "BG":           "#eef4fa",
        "SURFACE":      "#ffffff",
        "SURFACE2":     "#f8fafb",
        "FG":           "#1b2737",
        "MUTED":        "#5d7087",
        "PRIMARY":      "#15808d",
        "PRIMARY_SOFT": "#d9f0f3",
        "BORDER":       "#cfdbe7",
        "SUCCESS":      "#166534",
        "SUCCESS_BG":   "#dff7e8",
        "WARNING":      "#92400e",
        "WARNING_BG":   "#fef3c7",
        "ERROR":        "#991b1b",
        "ERROR_BG":     "#fee2e2",
        "INFO":         "#1d4ed8",
        "INFO_BG":      "#dbeafe",
        "ACCENT":       "#2dd4bf",
        "LOG_BG":       "#0d1117",
        "LOG_BORDER":   "#cfdbe7",
    },
    "dark": {
        "BG":           "#0d1117",
        "SURFACE":      "#1e293b",
        "SURFACE2":     "#334155",
        "FG":           "#f1f5f9",
        "MUTED":        "#94a3b8",
        "PRIMARY":      "#2dd4bf",
        "PRIMARY_SOFT": "#134e4a",
        "BORDER":       "#334155",
        "SUCCESS":      "#4ade75",
        "SUCCESS_BG":   "#052e16",
        "WARNING":      "#fbbf24",
        "WARNING_BG":   "#422006",
        "ERROR":        "#f87171",
        "ERROR_BG":     "#450a0a",
        "INFO":         "#60a5fa",
        "INFO_BG":      "#172554",
        "ACCENT":       "#2dd4bf",
        "LOG_BG":       "#161b22",
        "LOG_BORDER":   "#21262d",
    },
}

_current_theme = "light"

def set_theme(name):
    global _current_theme
    _current_theme = name
    for key, val in THEMES[name].items():
        globals()[key] = val

set_theme("light")

DPI = 150
FONT_FAMILY = "sans-serif"

# ─── Helpers ─────────────────────────────────────────────────────

def new_fig(w, h):
    fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=DPI)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax

def rrect(ax, x, y, w, h, fc=None, ec=None, lw=1, r=0.12, alpha=1):
    if fc is None: fc = SURFACE
    if ec is None: ec = BORDER
    box = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha)
    ax.add_patch(box)
    return box

def rect(ax, x, y, w, h, fc=None, ec=None, lw=1, alpha=1):
    if fc is None: fc = SURFACE
    if ec is None: ec = BORDER
    box = mpatches.Rectangle((x, y), w, h,
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha)
    ax.add_patch(box)
    return box

def text(ax, x, y, s, size=9, color=None, weight="normal", ha="left", va="top", family=FONT_FAMILY, alpha=1):
    if color is None: color = FG
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, family=family, alpha=alpha)

def line(ax, x1, y1, x2, y2, color=None, lw=1, ls="-"):
    if color is None: color = BORDER
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, zorder=5)

def badge(ax, x, y, label, bg=None, fg=None, w=1.8, h=0.3):
    if bg is None: bg = SUCCESS_BG
    if fg is None: fg = SUCCESS
    rrect(ax, x, y, w, h, fc=bg, ec=bg, r=0.08)
    text(ax, x + w/2, y + h/2, label, size=7, color=fg, ha="center", va="center", weight="bold")

def btn(ax, x, y, w, h, label, primary=False, size=8):
    fc = PRIMARY if primary else SURFACE
    ec = PRIMARY if primary else BORDER
    fg = "#ffffff" if primary else FG
    rrect(ax, x, y, w, h, fc=fc, ec=ec, r=0.08)
    text(ax, x + w/2, y + h/2, label, size=size, color=fg, ha="center", va="center", weight="bold")

def input_field(ax, x, y, w, h, value="", placeholder="", size=7):
    rrect(ax, x, y, w, h, fc=SURFACE2, ec=BORDER, r=0.06)
    if value:
        text(ax, x + 0.1, y + h/2, value, size=size, color=FG, va="center")
    elif placeholder:
        text(ax, x + 0.1, y + h/2, placeholder, size=size, color=MUTED, va="center")

def checkbox(ax, x, y, checked=False, label="", size=8):
    box = mpatches.Rectangle((x, y), 0.18, 0.18, facecolor=SURFACE, edgecolor=BORDER, linewidth=1)
    ax.add_patch(box)
    if checked:
        ax.text(x + 0.09, y + 0.09, "✓", fontsize=7, color=PRIMARY, ha="center", va="center", weight="bold")
    if label:
        text(ax, x + 0.28, y + 0.09, label, size=size, va="center")

def tab(ax, x, y, w, h, label, active=False, size=9):
    fc = PRIMARY if active else BG
    fg = "#ffffff" if active else FG
    rrect(ax, x, y, w, h, fc=fc, ec=BORDER if not active else PRIMARY, r=0.08)
    text(ax, x + w/2, y + h/2, label, size=size, color=fg, ha="center", va="center", weight="bold")

def progress_bar(ax, x, y, w, h, pct, color=None):
    if color is None: color = PRIMARY
    rrect(ax, x, y, w, h, fc=SURFACE2, ec=BORDER, r=0.04)
    if pct > 0:
        rrect(ax, x, y, w * pct / 100, h, fc=color, ec=color, r=0.04)

def section_title(ax, x, y, label, size=10):
    text(ax, x, y, label, size=size, color=FG, weight="bold")
    line(ax, x, y + 0.3, x + 20, y + 0.3, color=BORDER, lw=0.8)

def log_line(ax, x, y, timestamp, level, msg, size=7):
    colors = {"INFO": INFO, "WARN": WARNING, "ERROR": ERROR, "DEBUG": MUTED}
    c = colors.get(level, FG)
    text(ax, x, y, timestamp, size=size, color=MUTED, va="top")
    text(ax, x + 0.9, y, level, size=size, color=c, va="top", weight="bold")
    text(ax, x + 1.6, y, msg, size=size, color=FG, va="top")

def phase_marker(ax, x, y, w):
    line(ax, x, y, x + w, y, color=BORDER, lw=0.5, ls="--")

def save(fig, name):
    suffix = "" if _current_theme == "light" else "_dark"
    path = os.path.join(OUT, f"mockups/{name}{suffix}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.1, facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {name}{suffix}.png")

def draw_header(ax, active_tab):
    """Draw consistent header with logo, tabs, status badges, locale, theme toggle.
    active_tab: 'processing' | 'tools' | 'history'
    """
    rect(ax, 0, 0, 16, 0.7, fc=SURFACE, ec=BORDER)
    text(ax, 0.3, 0.35, "# tile_compile", size=12, color=PRIMARY, weight="bold", va="center")
    tab(ax, 5.0, 0.15, 2.0, 0.4, "Processing", active=active_tab == "processing")
    tab(ax, 7.2, 0.15, 1.5, 0.4, "Tools", active=active_tab == "tools")
    tab(ax, 8.9, 0.15, 1.9, 0.4, "History", active=active_tab == "history")
    badge(ax, 11.2, 0.2, "● Run ready", bg=SUCCESS_BG, fg=SUCCESS, w=1.5, h=0.3)
    badge(ax, 12.8, 0.2, "● Guardrails OK", bg=SUCCESS_BG, fg=SUCCESS, w=1.8, h=0.3)
    text(ax, 15.0, 0.35, "DE|EN", size=9, color=MUTED, va="center")
    icon = "☀" if _current_theme == "dark" else "☾"
    text(ax, 15.5, 0.35, icon, size=12, color=MUTED, va="center")

# ─── Mockup 01: Global Layout ────────────────────────────────────

def mock_global_layout():
    fig, ax = new_fig(16, 10)
    
    draw_header(ax, "processing")
    
    # Sub-tab bar
    rect(ax, 0, 0.7, 16, 0.5, fc=SURFACE2, ec=BORDER)
    tab(ax, 0.5, 0.8, 2.0, 0.35, "Input & Scan", active=True, size=8)
    tab(ax, 2.8, 0.8, 1.8, 0.35, "Parameter", active=False, size=8)
    tab(ax, 4.9, 0.8, 2.0, 0.35, "Run Monitor", active=False, size=8)
    
    # Content area
    rrect(ax, 0.3, 1.5, 15.4, 8.0, fc=SURFACE, ec=BORDER, r=0.15)
    text(ax, 8, 5.5, "Tab-Inhalt", size=14, color=MUTED, ha="center", va="center")
    
    # Footer
    rect(ax, 0, 9.5, 16, 0.5, fc=SURFACE, ec=BORDER)
    text(ax, 0.3, 9.75, "Bereit", size=8, color=MUTED, va="center")
    
    save(fig, "01_global_layout")

# ─── Mockup 02: Input & Scan ─────────────────────────────────────

def mock_input_scan():
    fig, ax = new_fig(16, 14)
    
    draw_header(ax, "processing")
    
    # Sub-tabs
    rect(ax, 0, 0.7, 16, 0.5, fc=SURFACE2, ec=BORDER)
    tab(ax, 0.5, 0.8, 2.0, 0.35, "Input & Scan", active=True, size=8)
    tab(ax, 2.8, 0.8, 1.8, 0.35, "Parameter", active=False, size=8)
    tab(ax, 4.9, 0.8, 2.0, 0.35, "Run Monitor", active=False, size=8)
    
    y = 1.5
    
    # Input section
    section_title(ax, 0.5, y, "Input")
    y += 0.5
    rrect(ax, 0.5, y, 15, 3.8, fc=SURFACE, ec=BORDER, r=0.1)
    
    fields = [
        ("Eingabeordner", "/data/M31/lights"),
        ("Dateimuster", "*.fits"),
        ("Ausgabeordner", "/data/runs"),
        ("Run Name", "M31_altaz_test"),
    ]
    for i, (label, val) in enumerate(fields):
        fy = y + 0.3 + i * 0.55
        text(ax, 0.8, fy, label, size=8, color=FG, va="center")
        input_field(ax, 3.5, fy - 0.12, 8, 0.35, value=val)
        btn(ax, 11.7, fy - 0.12, 0.6, 0.35, "[...]", size=7)
    
    text(ax, 0.8, y + 2.6, "→ Output: /data/runs/M31_altaz_test_20260620_213000", size=7, color=MUTED)
    
    fy = y + 3.1
    text(ax, 0.8, fy, "Frames Min", size=8, color=FG, va="center")
    input_field(ax, 3.5, fy - 0.12, 1.5, 0.35, value="30")
    text(ax, 5.5, fy, "Max. Frames", size=8, color=FG, va="center")
    input_field(ax, 7.0, fy - 0.12, 1.5, 0.35, value="0 (∞)")
    text(ax, 9.0, fy, "Sortierung", size=8, color=FG, va="center")
    input_field(ax, 10.5, fy - 0.12, 2, 0.35, value="numeric")
    text(ax, 13.0, fy, "Farbmodus", size=8, color=FG, va="center")
    input_field(ax, 14.0, fy - 0.12, 1.3, 0.35, value="MONO")
    
    y += 4.2
    
    # Run Queue
    section_title(ax, 0.5, y, "Run-Queue")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    # Table header
    cols = [("Filter", 0.8, 1.5), ("Input Dir", 2.5, 5), ("Pattern", 7.8, 2), ("Label", 10.0, 1.5), ("Aktiv", 11.8, 1)]
    for label, cx, cw in cols:
        text(ax, cx, y + 0.25, label, size=7, color=MUTED, weight="bold")
    line(ax, 0.8, y + 0.5, 15.2, y + 0.5, color=BORDER, lw=0.5)
    
    rows = [("L", "/data/M31/lights/L", "*.fits", "L", True),
            ("R", "/data/M31/lights/R", "*.fits", "R", True),
            ("G", "/data/M31/lights/G", "*.fits", "G", True),
            ("B", "/data/M31/lights/B", "*.fits", "B", True)]
    for i, (flt, d, p, lbl, actv) in enumerate(rows):
        ry = y + 0.7 + i * 0.4
        text(ax, 0.8, ry, f"[{flt}▼]", size=7, color=FG, va="top")
        text(ax, 2.5, ry, d, size=7, color=FG, va="top")
        text(ax, 7.8, ry, p, size=7, color=FG, va="top")
        text(ax, 10.0, ry, lbl, size=7, color=FG, va="top")
        checkbox(ax, 12.0, ry - 0.02, checked=actv, size=7)
    
    btn(ax, 14.5, y + 2.0, 0.8, 0.3, "+", primary=True, size=8)
    
    y += 2.9
    
    # Calibration
    section_title(ax, 0.5, y, "Kalibrierung")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.8, fc=SURFACE, ec=BORDER, r=0.1)
    
    cal_items = [("Bias", True, "/data/cals/bias"), ("Dark", True, "/data/cals/dark"), ("Flat", False, "")]
    for i, (name, checked, path) in enumerate(cal_items):
        cy = y + 0.3 + i * 0.45
        checkbox(ax, 0.8, cy, checked=checked, label=name, size=8)
        input_field(ax, 3.5, cy - 0.05, 8, 0.3, value=path)
        btn(ax, 11.7, cy - 0.05, 0.6, 0.3, "[...]", size=7)
    
    y += 2.2
    
    # Buttons
    btn(ax, 0.5, y, 2.5, 0.5, "Scan starten", primary=True, size=9)
    btn(ax, 3.2, y, 1.5, 0.5, "> Next", size=9)
    
    y += 1.0
    
    # Scan Result
    section_title(ax, 0.5, y, "Scan-Ergebnis")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    badge(ax, 0.8, y + 0.2, "✓ OK", bg=SUCCESS_BG, fg=SUCCESS, w=1.2, h=0.28)
    text(ax, 2.3, y + 0.35, "Frames: 325", size=8, color=FG, va="center")
    text(ax, 4.0, y + 0.35, "Color: OSC", size=8, color=FG, va="center")
    text(ax, 5.5, y + 0.35, "Bayer: RGGB", size=8, color=FG, va="center")
    text(ax, 7.2, y + 0.35, "Bildgröße: 4032×3024", size=8, color=FG, va="center")
    text(ax, 10.0, y + 0.35, "Fehler: 0", size=8, color=SUCCESS, va="center")
    text(ax, 11.5, y + 0.35, "Warnungen: 2", size=8, color=WARNING, va="center")
    
    text(ax, 0.8, y + 0.9, "! Frame 47: ungewöhnlicher Header-Wert (BAYERPAT=GBRG)", size=7, color=WARNING)
    text(ax, 0.8, y + 1.3, "! Frame 112: Dateigröße abweichend (0.3× Median)", size=7, color=WARNING)
    
    save(fig, "02_input_scan")

# ─── Mockup 03: Parameter Studio ─────────────────────────────────

def mock_parameter():
    fig, ax = new_fig(16, 12)
    
    draw_header(ax, "processing")
    
    # Sub-tabs
    rect(ax, 0, 0.7, 16, 0.5, fc=SURFACE2, ec=BORDER)
    tab(ax, 0.5, 0.8, 2.0, 0.35, "Input & Scan", active=False, size=8)
    tab(ax, 2.8, 0.8, 1.8, 0.35, "Parameter", active=True, size=8)
    tab(ax, 4.9, 0.8, 2.0, 0.35, "Run Monitor", active=False, size=8)
    
    # Pipeline mode badge + AI tab switch
    badge(ax, 7.2, 0.8, "Full Mode (≥200 Frames)", bg=PRIMARY_SOFT, fg=PRIMARY, w=3.0, h=0.35)
    tab(ax, 11.0, 0.8, 1.8, 0.35, "Parameter", active=True, size=8)
    tab(ax, 13.0, 0.8, 1.8, 0.35, "AI Empfehlung", active=False, size=8)
    
    y = 1.5
    
    # 3-column layout
    col1_x, col1_w = 0.3, 3.0
    col2_x, col2_w = 3.5, 7.0
    col3_x, col3_w = 10.7, 5.0
    
    # Column 1: Categories
    rrect(ax, col1_x, y, col1_w, 9.5, fc=SURFACE, ec=BORDER, r=0.1)
    text(ax, col1_x + 0.2, y + 0.2, "Q Suche...", size=8, color=MUTED)
    line(ax, col1_x + 0.2, y + 0.5, col1_x + col1_w - 0.2, y + 0.5, color=BORDER, lw=0.5)
    
    categories = ["Alle", "System", "Pipeline", "Input&Scan", "Linearity", "Calibration",
                  "Assumptions", "Normalization", "Registration", "Dithering", "Tile Denoise",
                  "Chroma D.", "Global Metr.", "Tile", "Local Metr.", "Synthetic",
                  "Debayer", "Astrometry", "BGE", "AQMH", "PCC", "Stacking",
                  "Runtime", "Validation", "Data"]
    for i, cat in enumerate(categories):
        cy = y + 0.7 + i * 0.35
        is_active = cat == "Registration"
        if is_active:
            rrect(ax, col1_x + 0.1, cy - 0.05, col1_w - 0.2, 0.3, fc=PRIMARY_SOFT, ec=PRIMARY_SOFT, r=0.04)
            text(ax, col1_x + 0.3, cy + 0.1, cat, size=7, color=PRIMARY, weight="bold", va="center")
        else:
            text(ax, col1_x + 0.3, cy + 0.1, cat, size=7, color=FG, va="center")
    
    # Column 2: Editor
    rrect(ax, col2_x, y, col2_w, 9.5, fc=SURFACE, ec=BORDER, r=0.1)
    text(ax, col2_x + 0.2, y + 0.2, "Registration", size=10, color=FG, weight="bold")
    line(ax, col2_x + 0.2, y + 0.5, col2_x + col2_w - 0.2, y + 0.5, color=BORDER, lw=0.5)
    
    params = [
        ("engine", "triangle_star_matching", "select"),
        ("allow_rotation", "true", "select"),
        ("transform_model", "affine", "select"),
        ("star_topk", "180", "input"),
        ("star_inlier_tol", "4.0", "input"),
        ("reject_cc_min", "0.25", "input"),
    ]
    for i, (name, val, ptype) in enumerate(params):
        py = y + 0.7 + i * 0.55
        text(ax, col2_x + 0.3, py, name, size=8, color=FG, va="center")
        input_field(ax, col2_x + 3.0, py - 0.12, 3.5, 0.35, value=val)
    
    # Preset + buttons
    by = y + 4.5
    text(ax, col2_x + 0.3, by, "Preset:", size=8, color=FG, va="center")
    input_field(ax, col2_x + 1.2, by - 0.12, 2.5, 0.35, value="M31.global")
    btn(ax, col2_x + 3.9, by - 0.12, 1.0, 0.35, "Apply", size=7)
    btn(ax, col2_x + 5.0, by - 0.12, 1.3, 0.35, "YAML Sync", size=7)
    
    by2 = y + 5.1
    btn(ax, col2_x + 0.3, by2, 1.3, 0.35, "Validate", primary=True, size=7)
    btn(ax, col2_x + 1.8, by2, 1.0, 0.35, "Save", size=7)
    
    # YAML Diff
    dy = y + 5.8
    rrect(ax, col2_x + 0.2, dy, col2_w - 0.4, 3.4, fc=LOG_BG, ec=BORDER, r=0.08)
    text(ax, col2_x + 0.4, dy + 0.2, "YAML Diff", size=8, color="#94a3b8", weight="bold")
    diffs = [
        ("- star_topk: 150", ERROR),
        ("+ star_topk: 180", SUCCESS),
        ("- engine: star_sim", ERROR),
        ("+ engine: triangle", SUCCESS),
    ]
    for i, (line_text, color) in enumerate(diffs):
        text(ax, col2_x + 0.4, dy + 0.6 + i * 0.35, line_text, size=7, color=color, family="monospace")
    
    # Column 3: Explain
    rrect(ax, col3_x, y, col3_w, 9.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, col3_x + 0.2, y + 0.2, "Explain", size=10, color=FG, weight="bold")
    line(ax, col3_x + 0.2, y + 0.5, col3_x + col3_w - 0.2, y + 0.5, color=BORDER, lw=0.5)
    
    explain_items = [
        ("Label", "registration.star_topk"),
        ("Kategorie", "registration"),
        ("Typ", "integer"),
        ("Default", "150"),
        ("Wertebereich", "50..500"),
        ("Phase", "REGISTRATION"),
    ]
    for i, (label, val) in enumerate(explain_items):
        ey = y + 0.7 + i * 0.4
        text(ax, col3_x + 0.3, ey, label, size=7, color=MUTED, va="center")
        text(ax, col3_x + 1.8, ey, val, size=8, color=FG, va="center", weight="bold")
    
    text(ax, col3_x + 0.3, y + 3.3, "Was macht der Parameter?", size=8, color=FG, weight="bold")
    text(ax, col3_x + 0.3, y + 3.7, "Anzahl Top-Sterne für\nMatching; mehr Robustheit\nbei schwieriger Registrierung.", size=7, color=MUTED)
    
    # Situation Assistant
    sy = y + 5.3
    rrect(ax, col3_x + 0.2, sy, col3_w - 0.4, 2.5, fc=SURFACE2, ec=BORDER, r=0.08)
    text(ax, col3_x + 0.4, sy + 0.2, "Situation Assistant", size=8, color=FG, weight="bold")
    
    sit_items = [("Alt/Az", True), ("Starke Rotation", False), ("Helle Sterne", True),
                 ("Wenige Frames", False), ("Starker Gradient", True)]
    for i, (label, checked) in enumerate(sit_items):
        cy = sy + 0.6 + i * 0.35
        checkbox(ax, col3_x + 0.4, cy, checked=checked, label=label, size=7)
    
    btn(ax, col3_x + 0.4, sy + 2.1, 1.5, 0.3, "Apply", primary=True, size=7)
    
    # Assumptions preview
    ay = y + 8.0
    rrect(ax, col3_x + 0.2, ay, col3_w - 0.4, 1.3, fc=SURFACE2, ec=BORDER, r=0.08)
    text(ax, col3_x + 0.4, ay + 0.15, "Assumptions", size=8, color=FG, weight="bold")
    text(ax, col3_x + 0.4, ay + 0.5, "frames_min: 30", size=7, color=FG)
    text(ax, col3_x + 0.4, ay + 0.8, "reduced: 200  Mode: Full", size=7, color=FG)
    
    save(fig, "03_parameter")

# ─── Mockup 04: AI Empfehlung ────────────────────────────────────

def mock_ai_empfehlung():
    fig, ax = new_fig(16, 13)
    
    draw_header(ax, "processing")
    
    # Sub-tabs
    rect(ax, 0, 0.7, 16, 0.5, fc=SURFACE2, ec=BORDER)
    tab(ax, 0.5, 0.8, 2.0, 0.35, "Input & Scan", active=False, size=8)
    tab(ax, 2.8, 0.8, 1.8, 0.35, "Parameter", active=True, size=8)
    tab(ax, 4.9, 0.8, 2.0, 0.35, "Run Monitor", active=False, size=8)
    
    badge(ax, 7.2, 0.8, "Full Mode (≥200 Frames)", bg=PRIMARY_SOFT, fg=PRIMARY, w=3.0, h=0.35)
    tab(ax, 11.0, 0.8, 1.8, 0.35, "Parameter", active=False, size=8)
    tab(ax, 13.0, 0.8, 1.8, 0.35, "AI Empfehlung", active=True, size=8)
    
    y = 1.5
    
    # Scan Context
    section_title(ax, 0.5, y, "Scan-Kontext (auto aus Scan)")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.8, fc=SURFACE, ec=BORDER, r=0.1)
    
    ctx_items = [("Mount", "EQ / Tracker"), ("Zielgröße", "Kompakt"), ("Kamera", "Consumer OSC")]
    for i, (label, val) in enumerate(ctx_items):
        cx = 0.8 + i * 4.5
        text(ax, cx, y + 0.3, label, size=8, color=MUTED, va="center")
        input_field(ax, cx + 1.5, y + 0.18, 2.5, 0.35, value=val)
    
    text(ax, 0.8, y + 0.9, "Kalibrierung:", size=8, color=MUTED, va="center")
    checkbox(ax, 2.5, y + 0.85, checked=True, label="Darks", size=7)
    checkbox(ax, 3.8, y + 0.85, checked=False, label="Flats", size=7)
    checkbox(ax, 5.1, y + 0.85, checked=False, label="Bias", size=7)
    text(ax, 0.8, y + 1.35, "Notizen:", size=8, color=MUTED, va="center")
    input_field(ax, 2.0, y + 1.23, 12, 0.35, value="Guiding 0.8\", M31, alt-az test")
    
    y += 2.2
    
    # Model & API Key
    section_title(ax, 0.5, y, "Modell & API-Key")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, 0.8, y + 0.3, "Provider", size=8, color=MUTED, va="center")
    input_field(ax, 2.0, y + 0.18, 2.5, 0.35, value="anthropic")
    text(ax, 5.0, y + 0.3, "Modell", size=8, color=MUTED, va="center")
    input_field(ax, 6.0, y + 0.18, 4, 0.35, value="claude-sonnet-4-20250514")
    text(ax, 10.5, y + 0.3, "API-Key", size=8, color=MUTED, va="center")
    input_field(ax, 11.5, y + 0.18, 2.5, 0.35, value="••••••••••")
    btn(ax, 14.2, y + 0.18, 1.0, 0.35, "Save", size=7)
    badge(ax, 0.8, y + 0.9, "✓ Modell verfügbar", bg=SUCCESS_BG, fg=SUCCESS, w=2.5, h=0.28)
    
    y += 1.9
    
    # Action buttons
    btn(ax, 0.5, y, 3.0, 0.5, "KI-Analyse erstellen", primary=True, size=9)
    btn(ax, 3.8, y, 3.5, 0.5, "Neu analysieren", size=9)
    text(ax, 7.6, y + 0.25, "Gespeicherte Analysen▼", size=8, color=MUTED, va="center")
    
    y += 1.0
    
    # Recommendations
    section_title(ax, 0.5, y, "Empfehlungen")
    y += 0.5
    rrect(ax, 0.5, y, 15, 5.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    recs = [
        ("☑", "registration.engine", "triangle_star_matching → hybrid_phase_ecc",
         "Alt-Az Mount erzeugt starke Feldrotation. hybrid_phase_ecc\nkompensiert Rotation + Translation gleichzeitig und ist robuster\nbei dithered frames.", "Risiko: niedrig", True),
        ("☑", "registration.star_topk", "180 → 250",
         "325 Frames mit Seeing-Schwankungen. Mehr Sterne geben\nrobusteren Match bei teilweise verwischten Frames.", "Risiko: minimal", True),
        ("☐", "bge.fit.method", "rbf → rbf (bereits optimal)",
         "RBF ist die beste Wahl für ausgedehnte Gradienten bei\nGalaxienfeldern. Keine Änderung nötig.", "Risiko: –", False),
        ("☑", "aqmh.cherry_pick.enabled", "false → true (k_frac=0.30, k_min=3)",
         "Bei 325 Frames mit Seeing-Schwankungen kann Cherry-Picking\ndie besten 30% auswählen und SQM deutlich verbessern.", "Risiko: mittel", True),
    ]
    
    for i, (check, param, change, reason, risk, _) in enumerate(recs):
        ry = y + 0.2 + i * 1.3
        text(ax, 0.8, ry, check, size=10, color=PRIMARY if "☑" in check else MUTED, va="top")
        text(ax, 1.3, ry, param, size=8, color=FG, weight="bold", va="top")
        text(ax, 4.5, ry, change, size=7, color=PRIMARY if "☑" in check else MUTED, va="top")
        text(ax, 0.8, ry + 0.35, reason, size=7, color=MUTED, va="top")
        risk_color = WARNING if "mittel" in risk else SUCCESS if "niedrig" in risk or "minimal" in risk else MUTED
        text(ax, 13.5, ry, risk, size=7, color=risk_color, va="top")
        if i < len(recs) - 1:
            line(ax, 0.8, ry + 1.1, 15.2, ry + 1.1, color=BORDER, lw=0.4)
    
    y += 5.8
    
    # Apply buttons
    btn(ax, 0.5, y, 3.0, 0.5, "Ausgewählte anwenden (3)", primary=True, size=9)
    btn(ax, 3.8, y, 2.0, 0.5, "Alle anwenden", size=9)
    btn(ax, 6.0, y, 1.8, 0.5, "Verwerfen", size=9)
    
    y += 0.8
    text(ax, 0.5, y, "▸ KI-Datenverkehr (ausgeblendet)", size=8, color=MUTED)
    
    save(fig, "04_ai_empfehlung")

# ─── Mockup 05: Run Monitor ──────────────────────────────────────

def mock_run_monitor():
    fig, ax = new_fig(16, 15)
    
    draw_header(ax, "processing")
    
    # Sub-tabs
    rect(ax, 0, 0.7, 16, 0.5, fc=SURFACE2, ec=BORDER)
    tab(ax, 0.5, 0.8, 2.0, 0.35, "Input & Scan", active=False, size=8)
    tab(ax, 2.8, 0.8, 1.8, 0.35, "Parameter", active=False, size=8)
    tab(ax, 4.9, 0.8, 2.0, 0.35, "Run Monitor", active=True, size=8)
    
    y = 1.5
    
    # Run Control
    section_title(ax, 0.5, y, "Run-Steuerung")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.2, fc=SURFACE, ec=BORDER, r=0.1)
    text(ax, 0.8, y + 0.3, "Run: M31_altaz_test_20260620_213000", size=9, color=FG, weight="bold", va="center")
    btn(ax, 0.8, y + 0.6, 1.8, 0.4, "> Run starten", primary=True, size=8)
    btn(ax, 2.8, y + 0.6, 1.3, 0.4, "[x] Stop", size=8)
    btn(ax, 4.3, y + 0.6, 2.2, 0.4, "Run-Ordner öffnen", size=8)
    text(ax, 7.0, y + 0.8, "! Validierung: Config nicht validiert – Run blockiert", size=7, color=WARNING, va="center")
    
    y += 1.6
    
    # Phases
    section_title(ax, 0.5, y, "Phasen")
    y += 0.5
    rrect(ax, 0.5, y, 15, 5.0, fc=SURFACE, ec=BORDER, r=0.1)
    
    phases = [
        ("[v]", "SCAN", "325 frames, 3.2s", 100, SUCCESS),
        ("[v]", "CALIBRATION", "325 frames, 12.4s", 100, SUCCESS),
        ("[v]", "REGISTRATION", "325/325, 26.1s", 100, SUCCESS),
        ("[v]", "NORMALIZATION", "done, 2.1s", 100, SUCCESS),
        ("[~]", "AQMH", "180/325, 45.3s", 55, PRIMARY),
        ("[ ]", "STACKING", "waiting", 0, MUTED),
        ("[ ]", "ASTROMETRY", "waiting", 0, MUTED),
        ("[ ]", "BGE", "waiting", 0, MUTED),
        ("[ ]", "PCC", "waiting", 0, MUTED),
        ("[ ]", "HYPERMETRIC_STRETCH", "waiting", 0, MUTED),
    ]
    
    for i, (icon, name, detail, pct, color) in enumerate(phases):
        py = y + 0.2 + i * 0.45
        text(ax, 0.8, py, icon, size=9, va="top")
        text(ax, 1.5, py, name, size=8, color=FG, weight="bold", va="top")
        text(ax, 5.0, py, detail, size=7, color=MUTED, va="top")
        progress_bar(ax, 8.0, py + 0.05, 6.5, 0.2, pct, color=color)
        text(ax, 14.8, py, f"{pct}%", size=7, color=color, va="top", weight="bold")
    
    text(ax, 0.8, y + 4.7, "! AQMH Cherry-Pick aktiv: 180/325 frames (k_frac=0.30)", size=7, color=WARNING)
    
    y += 5.4
    
    # Live Log
    section_title(ax, 0.5, y, "Live Log")
    y += 0.5
    rrect(ax, 0.5, y, 15, 4.5, fc=LOG_BG, ec=BORDER, r=0.1)
    
    # Toolbar
    text(ax, 0.8, y + 0.2, "[All▼]", size=7, color="#94a3b8", va="top")
    text(ax, 1.8, y + 0.2, "[Q Suche...]", size=7, color="#7d8590", va="top")
    text(ax, 12.5, y + 0.2, "[[ ] Pause]", size=7, color="#94a3b8", va="top")
    text(ax, 14.0, y + 0.2, "[v Export]", size=7, color="#94a3b8", va="top")
    line(ax, 0.8, y + 0.5, 15.2, y + 0.5, color=LOG_BORDER, lw=0.5)
    
    log_entries = [
        ("21:15:32", "INFO", "Phase SCAN started"),
        ("21:15:33", "INFO", "Found 325 frames in /data/M31"),
        ("21:15:34", "INFO", "Color mode: OSC, Bayer: RGGB"),
        ("21:15:35", "INFO", "Phase SCAN completed (3.2s)"),
        ("PHASE", "", "─" * 60),
        ("21:15:35", "INFO", "Phase CALIBRATION started"),
        ("21:15:47", "INFO", "Phase CALIBRATION completed (12.4s)"),
        ("PHASE", "", "─" * 60),
        ("21:15:47", "INFO", "Phase REGISTRATION started"),
        ("21:15:52", "WARN", "Frame 47: low CC=0.31, sequential"),
        ("21:16:01", "INFO", "Frame 112: hot pixel detected, clamped"),
        ("21:16:13", "INFO", "Phase REGISTRATION completed (26.1s)"),
        ("PHASE", "", "─" * 60),
        ("21:16:13", "INFO", "Phase AQMH started"),
        ("21:16:58", "INFO", "AQMH: processing window 180/325"),
        ("21:17:12", "WARN", "AQMH: cherry-pick selected 98 frames"),
    ]
    
    log_colors = {"INFO": "#58a6ff", "WARN": "#d29922", "ERROR": "#f85149", "DEBUG": "#7d8590"}
    
    for i, (ts, level, msg) in enumerate(log_entries):
        ly = y + 0.7 + i * 0.24
        if ts == "PHASE":
            line(ax, 0.8, ly + 0.08, 15.2, ly + 0.08, color=LOG_BORDER, lw=0.5)
        else:
            text(ax, 0.8, ly, ts, size=6, color="#7d8590", va="top", family="monospace")
            text(ax, 2.0, ly, level, size=6, color=log_colors.get(level, "#e6edf3"), va="top", weight="bold", family="monospace")
            text(ax, 3.2, ly, msg, size=6, color="#e6edf3", va="top", family="monospace")
    
    y += 5.0
    
    # Stats & Report
    section_title(ax, 0.5, y, "Stats & Report")
    y += 0.5
    rrect(ax, 0.5, y, 15, 0.8, fc=SURFACE, ec=BORDER, r=0.1)
    btn(ax, 0.8, y + 0.2, 1.8, 0.4, "Generate Stats", size=8)
    btn(ax, 2.8, y + 0.2, 2.2, 0.4, "Open Stats Folder", size=8)
    btn(ax, 5.2, y + 0.2, 1.8, 0.4, "Open Report", size=8)
    badge(ax, 7.3, y + 0.25, "✓ Stats generated", bg=SUCCESS_BG, fg=SUCCESS, w=2.5, h=0.3)
    
    y += 1.2
    
    # Resume & Config
    section_title(ax, 0.5, y, "Resume & Config-Revision")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, 0.8, y + 0.3, "Config Revision", size=8, color=MUTED, va="center")
    input_field(ax, 2.5, y + 0.18, 2, 0.35, value="rev_003")
    btn(ax, 4.7, y + 0.18, 1.3, 0.35, "Laden", size=7)
    text(ax, 6.5, y + 0.3, "Template", size=8, color=MUTED, va="center")
    input_field(ax, 7.5, y + 0.18, 2, 0.35, value="M31.global")
    btn(ax, 9.7, y + 0.18, 1.3, 0.35, "Laden", size=7)
    
    rrect(ax, 0.8, y + 0.7, 9, 1.5, fc=LOG_BG, ec=BORDER, r=0.06)
    yaml_lines = [
        "pipeline:",
        "  method: aqmh",
        "  resume_from: STACKING",
        "registration:",
        "  engine: hybrid_phase_ecc",
        "  star_topk: 250",
    ]
    for i, yl in enumerate(yaml_lines):
        text(ax, 1.0, y + 0.85 + i * 0.22, yl, size=6, color="#e6edf3", family="monospace")
    
    btn(ax, 10.5, y + 1.5, 2.0, 0.4, "Resume starten", primary=True, size=8)
    
    y += 2.9
    
    # Artifacts
    section_title(ax, 0.5, y, "Artefakte")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.0, fc=SURFACE, ec=BORDER, r=0.1)
    
    artifacts = [
        ("[D]", "outputs/", ""),
        ("[F]", "stack_M31.fits", "(45.2 MB)"),
        ("[F]", "stack_M31_weight.fits", "(12.1 MB)"),
        ("[D]", "artifacts/", ""),
        ("[F]", "stats.json", "(8.4 KB)"),
        ("[F]", "report.html", "(124 KB)"),
    ]
    for i, (icon, name, size) in enumerate(artifacts):
        ay = y + 0.2 + i * 0.3
        text(ax, 0.8, ay, f"{icon} {name}", size=7, color=FG, va="top")
        text(ax, 6.0, ay, size, size=7, color=MUTED, va="top")
    
    save(fig, "05_run_monitor")

# ─── Mockup 06: Run History ──────────────────────────────────────

def mock_run_history():
    fig, ax = new_fig(16, 11)
    
    draw_header(ax, "history")

    # Sub-tabs
    rect(ax, 0, 0.7, 16, 0.5, fc=SURFACE2, ec=BORDER)
    tab(ax, 0.5, 0.8, 2.0, 0.35, "Run History", active=True, size=8)
    # No sub-tabs for History (single sub-tab)
    
    y = 1.5
    
    text(ax, 0.5, y, "Quelle: /data/runs", size=9, color=FG, weight="bold")
    btn(ax, 14, y - 0.05, 1.5, 0.35, "[~] Refresh", size=7)
    
    y += 0.5
    
    # Run List
    section_title(ax, 0.5, y, "Run-Liste")
    y += 0.5
    rrect(ax, 0.5, y, 15, 3.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    runs = [
        ("[AQMH]", "[~] RUNNING", "20260620_213000", "M31_altaz_test", True),
        ("[AQMH]", "[v] OK", "20260306_184430", "IC434_test", False),
        ("[AQMH]", "[v] OK", "20260305_201230", "NGC7000_v2", False),
        ("[TCC]", "[x] ERROR", "20260305_231155", "NGC7000", False),
        ("[AQMH]", "[v] OK", "20260304_182010", "M42_widefield", False),
        ("[AQMH]", "[x] STOPPED", "20260303_120000", "test_run", False),
    ]
    
    for i, (method, status, run_id, name, selected) in enumerate(runs):
        ry = y + 0.2 + i * 0.5
        if selected:
            rrect(ax, 0.7, ry - 0.05, 14.5, 0.45, fc=PRIMARY_SOFT, ec=PRIMARY_SOFT, r=0.04)
        text(ax, 0.8, ry, method, size=7, color=FG, va="top")
        status_color = SUCCESS if "OK" in status else ERROR if "ERROR" in status else WARNING if "STOPPED" in status else PRIMARY
        text(ax, 2.2, ry, status, size=7, color=status_color, va="top", weight="bold")
        text(ax, 4.5, ry, run_id, size=7, color=FG, va="top", family="monospace")
        text(ax, 8.0, ry, name, size=7, color=FG, va="top")
        text(ax, 14.5, ry, "[→]", size=7, color=PRIMARY, va="top")
    
    y += 4.0
    
    # Selected Run
    section_title(ax, 0.5, y, "Ausgewählter Run")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.8, fc=SURFACE, ec=BORDER, r=0.1)
    
    details = [
        ("Run ID", "M31_altaz_test_20260620_213000"),
        ("Status", "[~] RUNNING"),
        ("Phase", "AQMH (55%)"),
        ("Artefakte", "12 Dateien"),
        ("Report", "Nicht verfügbar (Run läuft)"),
        ("Run-Ordner", "/data/runs/M31_altaz_test_20260620_213000"),
    ]
    for i, (label, val) in enumerate(details):
        dy = y + 0.2 + i * 0.35
        text(ax, 0.8, dy, label, size=8, color=MUTED, va="center")
        text(ax, 3.0, dy, val, size=8, color=FG, va="center")
    
    progress_bar(ax, 0.8, y + 2.3, 10, 0.25, 55, color=PRIMARY)
    text(ax, 11.0, y + 2.35, "55%", size=8, color=PRIMARY, va="center", weight="bold")
    
    btn(ax, 12.5, y + 0.2, 2.5, 0.35, "Als Current Run", size=7)
    btn(ax, 12.5, y + 0.65, 2.5, 0.35, "Generate Stats", size=7)
    btn(ax, 12.5, y + 1.1, 2.5, 0.35, "Report öffnen", size=7)
    btn(ax, 12.5, y + 1.55, 2.5, 0.35, "Eintrag löschen", size=7)
    
    y += 3.2
    
    # Comparison
    section_title(ax, 0.5, y, "Run-Vergleich")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.0, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, 0.8, y + 0.2, "Vergleichs-Run", size=8, color=MUTED, va="center")
    input_field(ax, 2.5, y + 0.08, 4, 0.35, value="IC434_test_20260306")
    
    # Side by side
    rrect(ax, 0.8, y + 0.6, 6.5, 1.2, fc=SURFACE2, ec=BORDER, r=0.06)
    text(ax, 1.0, y + 0.7, "Run A: M31_altaz_test", size=7, color=FG, weight="bold", va="top")
    text(ax, 1.0, y + 0.95, "Status: RUNNING  Phase: AQMH 55%", size=6, color=MUTED, va="top")
    text(ax, 1.0, y + 1.2, "Frames: 325  Artefakte: 12", size=6, color=MUTED, va="top")
    
    rrect(ax, 8.0, y + 0.6, 6.5, 1.2, fc=SURFACE2, ec=BORDER, r=0.06)
    text(ax, 8.2, y + 0.7, "Run B: IC434_test", size=7, color=FG, weight="bold", va="top")
    text(ax, 8.2, y + 0.95, "Status: OK  Phase: DONE", size=6, color=MUTED, va="top")
    text(ax, 8.2, y + 1.2, "Frames: 180  Artefakte: 8", size=6, color=MUTED, va="top")
    
    save(fig, "06_run_history")

# ─── Mockup 07: Raw Stack ────────────────────────────────────────

def mock_raw_stack():
    fig, ax = new_fig(16, 14)
    
    draw_header(ax, "tools")

    # Sub-tabs
    rect(ax, 0, 0.7, 16, 0.5, fc=SURFACE2, ec=BORDER)
    tab(ax, 0.5, 0.8, 2.0, 0.35, "Raw Stack", active=True, size=8)
    tab(ax, 2.8, 0.8, 2.0, 0.35, "Astrometry", active=False, size=8)
    tab(ax, 5.1, 0.8, 1.5, 0.35, "PCC", active=False, size=8)
    
    y = 1.5
    
    # Input
    section_title(ax, 0.5, y, "Input")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    fields = [("Eingabeordner", "/data/M31/lights"), ("Dateimuster", "*.fits"),
              ("Ausgabeordner", "/data/runs"), ("Run Name", "M31_raw_stack")]
    for i, (label, val) in enumerate(fields):
        fy = y + 0.2 + i * 0.45
        text(ax, 0.8, fy, label, size=8, color=FG, va="center")
        input_field(ax, 3.5, fy - 0.12, 8, 0.35, value=val)
        btn(ax, 11.7, fy - 0.12, 0.6, 0.35, "[...]", size=7)
    
    fy = y + 2.05
    text(ax, 0.8, fy, "Farbmodus", size=8, color=FG, va="center")
    input_field(ax, 2.0, fy - 0.12, 1.5, 0.35, value="OSC")
    text(ax, 4.0, fy, "Bayer", size=8, color=FG, va="center")
    input_field(ax, 4.8, fy - 0.12, 1.5, 0.35, value="auto")
    text(ax, 7.0, fy, "Frames Min", size=8, color=FG, va="center")
    input_field(ax, 8.3, fy - 0.12, 1.2, 0.35, value="30")
    
    y += 2.9
    
    # Calibration
    section_title(ax, 0.5, y, "Kalibrierung")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.5, fc=SURFACE, ec=BORDER, r=0.1)
    cal = [("Bias", True, "/data/cals/bias"), ("Dark", True, "/data/cals/dark"), ("Flat", False, "")]
    for i, (name, checked, path) in enumerate(cal):
        cy = y + 0.2 + i * 0.4
        checkbox(ax, 0.8, cy, checked=checked, label=name, size=8)
        input_field(ax, 3.5, cy - 0.05, 8, 0.3, value=path)
        btn(ax, 11.7, cy - 0.05, 0.6, 0.3, "[...]", size=7)
    
    y += 1.9
    
    # Quality Filtering
    section_title(ax, 0.5, y, "Quality Filtering")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.8, fc=SURFACE, ec=BORDER, r=0.1)
    checkbox(ax, 0.8, y + 0.2, checked=True, label="Quality filtering aktivieren", size=8)
    
    qf = [("Min. FWHM", "1.5"), ("Max. FWHM", "8.0"), ("Min. Ecc.", "0.00"), ("Max. Ecc.", "0.85"), ("Min. SNR", "10")]
    for i, (label, val) in enumerate(qf):
        qx = 0.8 + i * 2.9
        text(ax, qx, y + 0.8, label, size=7, color=MUTED, va="center")
        input_field(ax, qx + 1.3, y + 0.68, 1.2, 0.3, value=val)
    
    y += 2.2
    
    # Stack Parameters
    section_title(ax, 0.5, y, "Stack Parameters")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.2, fc=SURFACE, ec=BORDER, r=0.1)
    text(ax, 0.8, y + 0.2, "Stack Method", size=8, color=FG, va="center")
    input_field(ax, 2.5, y + 0.08, 2.5, 0.35, value="sigma_clip")
    text(ax, 5.5, y + 0.2, "Sigma low/high", size=8, color=FG, va="center")
    input_field(ax, 7.0, y + 0.08, 1, 0.35, value="3.0")
    text(ax, 8.2, y + 0.2, "/", size=8, color=FG, va="center")
    input_field(ax, 8.5, y + 0.08, 1, 0.35, value="3.0")
    checkbox(ax, 10.0, y + 0.15, checked=True, label="Weighted stacking", size=8)
    
    y += 1.6
    
    # Postprocess
    section_title(ax, 0.5, y, "Postprocess")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.2, fc=SURFACE, ec=BORDER, r=0.1)
    pp = ["Astrometry (Plate Solve)", "BGE (Background Extraction)", "PCC (Color Calibration)", "HyperMetric Stretch"]
    for i, item in enumerate(pp):
        px = 0.8 + i * 3.7
        checkbox(ax, px, y + 0.3, checked=True, label=item, size=7)
    
    y += 1.6
    
    # Buttons
    btn(ax, 0.5, y, 3.0, 0.5, "> Preprocessing starten", primary=True, size=9)
    btn(ax, 3.8, y, 2.0, 0.5, "[x] Abbrechen", size=9)
    
    y += 1.0
    
    # Status
    section_title(ax, 0.5, y, "Status")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.2, fc=SURFACE, ec=BORDER, r=0.1)
    text(ax, 0.8, y + 0.2, "Job: preprocessing_abc123", size=8, color=FG, va="center")
    text(ax, 0.8, y + 0.5, "Status: [~] Running  Phase: Stacking (4/5)", size=8, color=PRIMARY, va="center")
    progress_bar(ax, 0.8, y + 0.8, 14, 0.25, 80, color=PRIMARY)
    
    y += 1.6
    
    # Log
    section_title(ax, 0.5, y, "Log")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.8, fc=LOG_BG, ec=BORDER, r=0.1)
    
    log_entries = [
        ("21:30:01", "INFO", "Preprocessing started"),
        ("21:30:05", "INFO", "Calibration: applying master bias..."),
        ("21:30:12", "INFO", "Calibration: applying master dark..."),
        ("21:30:25", "INFO", "Quality: 312/325 frames passed"),
        ("21:30:30", "INFO", "Stacking: 156/325 frames"),
    ]
    for i, (ts, level, msg) in enumerate(log_entries):
        ly = y + 0.15 + i * 0.28
        text(ax, 0.8, ly, ts, size=6, color="#7d8590", va="top", family="monospace")
        text(ax, 2.0, ly, level, size=6, color="#58a6ff", va="top", weight="bold", family="monospace")
        text(ax, 3.2, ly, msg, size=6, color="#e6edf3", va="top", family="monospace")
    
    save(fig, "07_raw_stack")

# ─── Mockup 08: Astrometry ───────────────────────────────────────

def mock_astrometry():
    fig, ax = new_fig(16, 12)
    
    draw_header(ax, "tools")

    # Sub-tabs
    rect(ax, 0, 0.7, 16, 0.5, fc=SURFACE2, ec=BORDER)
    tab(ax, 0.5, 0.8, 2.0, 0.35, "Raw Stack", active=False, size=8)
    tab(ax, 2.8, 0.8, 2.0, 0.35, "Astrometry", active=True, size=8)
    tab(ax, 5.1, 0.8, 1.5, 0.35, "PCC", active=False, size=8)
    
    y = 1.5
    
    # ASTAP Setup
    section_title(ax, 0.5, y, "ASTAP Setup")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.2, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, 0.8, y + 0.2, "ASTAP CLI", size=8, color=FG, va="center")
    input_field(ax, 2.5, y + 0.08, 9, 0.35, value="/usr/local/bin/astap")
    btn(ax, 11.7, y + 0.08, 0.6, 0.35, "[...]", size=7)
    
    text(ax, 0.8, y + 0.6, "ASTAP Data", size=8, color=FG, va="center")
    input_field(ax, 2.5, y + 0.48, 9, 0.35, value="/media/data/Astro/astap")
    btn(ax, 11.7, y + 0.48, 0.6, 0.35, "[...]", size=7)
    
    badge(ax, 0.8, y + 1.0, "✓ ASTAP gefunden (v2.1.4)", bg=SUCCESS_BG, fg=SUCCESS, w=3.0, h=0.28)
    btn(ax, 4.2, y + 1.0, 1.8, 0.3, "Detect ASTAP", size=7)
    btn(ax, 6.2, y + 1.0, 2.5, 0.3, "Install ASTAP CLI", size=7)
    text(ax, 9.0, y + 1.1, "Download-Status: ✓ Installation abgeschlossen", size=7, color=SUCCESS, va="center")
    
    y += 2.6
    
    # Star Database
    section_title(ax, 0.5, y, "Star Database")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.8, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, 0.8, y + 0.2, "Catalog", size=8, color=FG, va="center")
    input_field(ax, 2.0, y + 0.08, 5, 0.35, value="D50 (~800 MB, empfohlen)")
    text(ax, 7.5, y + 0.2, "Quelle: SourceForge ASTAP Star Databases", size=7, color=MUTED, va="center")
    
    btn(ax, 0.8, y + 0.6, 2.0, 0.35, "Download Catalog", primary=True, size=7)
    btn(ax, 3.0, y + 0.6, 2.0, 0.35, "Cancel Download", size=7)
    badge(ax, 5.3, y + 0.65, "✓ D50 installiert", bg=SUCCESS_BG, fg=SUCCESS, w=2.5, h=0.28)
    
    y += 2.2
    
    # Plate Solve
    section_title(ax, 0.5, y, "Plate Solve")
    y += 0.5
    rrect(ax, 0.5, y, 15, 4.0, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, 0.8, y + 0.2, "FITS File", size=8, color=FG, va="center")
    input_field(ax, 2.5, y + 0.08, 10, 0.35, value="/data/runs/M31/outputs/stack_M31.fits")
    btn(ax, 12.7, y + 0.08, 0.6, 0.35, "[...]", size=7)
    
    btn(ax, 0.8, y + 0.6, 1.5, 0.35, "Browse", size=7)
    btn(ax, 2.5, y + 0.6, 1.5, 0.35, "Solve", primary=True, size=7)
    btn(ax, 4.2, y + 0.6, 1.8, 0.35, "Save Solved", size=7)
    
    # WCS Results
    rrect(ax, 0.8, y + 1.2, 14, 2.5, fc=SURFACE2, ec=BORDER, r=0.08)
    text(ax, 1.0, y + 1.35, "WCS Results", size=8, color=FG, weight="bold", va="top")
    
    wcs = [("RA (J2000)", "00h 42m 44s"), ("Dec (J2000)", "+41° 16' 09\""),
           ("Pixel Scale", "1.85 \"/px"), ("Rotation", "-12.3°"), ("FOV", "2.1° × 1.6°")]
    for i, (label, val) in enumerate(wcs):
        wx = 1.0 + (i % 3) * 4.5
        wy = y + 1.7 + (i // 3) * 0.5
        text(ax, wx, wy, label, size=7, color=MUTED, va="center")
        text(ax, wx + 1.8, wy, val, size=8, color=FG, va="center", weight="bold")
    
    y += 4.5
    
    # Log
    section_title(ax, 0.5, y, "Log")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.5, fc=LOG_BG, ec=BORDER, r=0.1)
    
    log_entries = [
        ("21:35:01", "INFO", "ASTAP solve started"),
        ("21:35:03", "INFO", "Reading star database D50..."),
        ("21:35:08", "INFO", "Pattern matching..."),
        ("21:35:12", "INFO", "Solution found: RA=00:42:44 Dec=+41:16:09"),
    ]
    for i, (ts, level, msg) in enumerate(log_entries):
        ly = y + 0.15 + i * 0.28
        text(ax, 0.8, ly, ts, size=6, color="#7d8590", va="top", family="monospace")
        text(ax, 2.0, ly, level, size=6, color="#58a6ff", va="top", weight="bold", family="monospace")
        text(ax, 3.2, ly, msg, size=6, color="#e6edf3", va="top", family="monospace")
    
    save(fig, "08_astrometry")

# ─── Mockup 09: PCC ──────────────────────────────────────────────

def mock_pcc():
    fig, ax = new_fig(16, 13)
    
    draw_header(ax, "tools")

    # Sub-tabs
    rect(ax, 0, 0.7, 16, 0.5, fc=SURFACE2, ec=BORDER)
    tab(ax, 0.5, 0.8, 2.0, 0.35, "Raw Stack", active=False, size=8)
    tab(ax, 2.8, 0.8, 2.0, 0.35, "Astrometry", active=False, size=8)
    tab(ax, 5.1, 0.8, 1.5, 0.35, "PCC", active=True, size=8)
    
    y = 1.5
    
    # Input
    section_title(ax, 0.5, y, "Input")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.8, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, 0.8, y + 0.2, "RGB FITS", size=8, color=FG, va="center")
    input_field(ax, 2.5, y + 0.08, 10, 0.35, value="/data/runs/M31/outputs/stack_M31.fits")
    btn(ax, 12.7, y + 0.08, 0.6, 0.35, "[...]", size=7)
    
    text(ax, 0.8, y + 0.6, "WCS File", size=8, color=FG, va="center")
    input_field(ax, 2.5, y + 0.48, 10, 0.35, value="/data/runs/M31/outputs/stack_M31.wcs")
    btn(ax, 12.7, y + 0.48, 0.6, 0.35, "[...]", size=7)
    
    text(ax, 0.8, y + 1.0, "i Wenn RGB/WCS aus einem Run stammen, werden PCC-Parameter automatisch aus der config.yaml übernommen.", size=7, color=INFO)
    
    y += 2.2
    
    # Catalog Source
    section_title(ax, 0.5, y, "Catalog Source")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.2, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, 0.8, y + 0.2, "Source", size=8, color=FG, va="center")
    input_field(ax, 2.0, y + 0.08, 3, 0.35, value="siril")
    text(ax, 5.5, y + 0.2, "(Siril: lokale Gaia-DR3-XP Chunks)", size=7, color=MUTED, va="center")
    
    badge(ax, 0.8, y + 0.6, "✓ Alle 48 Chunks installiert", bg=SUCCESS_BG, fg=SUCCESS, w=3.0, h=0.28)
    text(ax, 4.2, y + 0.7, "Missing: 0", size=7, color=SUCCESS, va="center")
    
    text(ax, 0.8, y + 1.1, "Catalog Dir", size=8, color=FG, va="center")
    input_field(ax, 2.5, y + 0.98, 8, 0.35, value="/media/data/Astro/siril_catalog")
    btn(ax, 10.7, y + 0.98, 0.6, 0.35, "[...]", size=7)
    
    btn(ax, 0.8, y + 1.5, 2.0, 0.35, "Browse Catalog", size=7)
    btn(ax, 3.0, y + 1.5, 2.0, 0.35, "Download Missing", size=7)
    btn(ax, 5.2, y + 1.5, 1.5, 0.35, "Cancel", size=7)
    btn(ax, 6.9, y + 1.5, 2.0, 0.35, "Check Online", size=7)
    
    y += 2.6
    
    # PCC Parameters
    section_title(ax, 0.5, y, "PCC Parameters")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    params = [
        ("mag_limit", "14.0"), ("mag_bright_limit", "6.0"), ("min_stars", "10"),
        ("sigma_clip", "2.5"), ("aperture_radius_px", "8.0"), ("annulus_inner_px", "12.0"),
        ("annulus_outer_px", "18.0"), ("k_max", "3.2"), ("chroma_strength", "1.0"),
    ]
    for i, (name, val) in enumerate(params):
        px = 0.8 + (i % 3) * 5.0
        py = y + 0.2 + (i // 3) * 0.55
        text(ax, px, py, name, size=7, color=MUTED, va="center")
        input_field(ax, px + 2.2, py - 0.12, 2, 0.35, value=val)
    
    text(ax, 0.8, y + 1.9, "apply_attenuation", size=7, color=MUTED, va="center")
    input_field(ax, 3.0, y + 1.78, 1.5, 0.35, value="false")
    text(ax, 5.0, y + 1.9, "bg_neutralization", size=7, color=MUTED, va="center")
    input_field(ax, 7.2, y + 1.78, 1.5, 0.35, value="auto")
    
    y += 2.9
    
    # Buttons
    btn(ax, 0.5, y, 2.0, 0.5, "Run PCC", primary=True, size=9)
    btn(ax, 2.8, y, 2.0, 0.5, "Save Corrected", size=9)
    
    y += 1.0
    
    # Result
    section_title(ax, 0.5, y, "Result")
    y += 0.5
    rrect(ax, 0.5, y, 15, 2.5, fc=SURFACE, ec=BORDER, r=0.1)
    
    text(ax, 0.8, y + 0.2, "Stars matched", size=8, color=MUTED, va="center")
    text(ax, 2.5, y + 0.2, "142", size=8, color=FG, va="center", weight="bold")
    text(ax, 4.0, y + 0.2, "Stars used", size=8, color=MUTED, va="center")
    text(ax, 5.5, y + 0.2, "89", size=8, color=FG, va="center", weight="bold")
    text(ax, 7.0, y + 0.2, "Residual RMS", size=8, color=MUTED, va="center")
    text(ax, 8.8, y + 0.2, "0.018 mag", size=8, color=FG, va="center", weight="bold")
    
    text(ax, 0.8, y + 0.6, "Color Matrix:", size=8, color=FG, weight="bold", va="center")
    
    rrect(ax, 0.8, y + 0.9, 6, 1.3, fc=LOG_BG, ec=BORDER, r=0.06)
    matrix = ["R:  1.034   -0.012    0.000", "G:  0.003    0.987    0.002", "B: -0.001    0.008    1.124"]
    for i, ml in enumerate(matrix):
        text(ax, 1.0, y + 1.05 + i * 0.35, ml, size=7, color="#e6edf3", family="monospace")
    
    y += 3.0
    
    # Log
    section_title(ax, 0.5, y, "Log")
    y += 0.5
    rrect(ax, 0.5, y, 15, 1.5, fc=LOG_BG, ec=BORDER, r=0.1)
    
    log_entries = [
        ("21:40:01", "INFO", "PCC started"),
        ("21:40:03", "INFO", "Loading catalog: siril Gaia-DR3-XP"),
        ("21:40:08", "INFO", "Matched 142 stars, using 89 for fit"),
        ("21:40:10", "INFO", "PCC completed. RMS=0.018 mag"),
    ]
    for i, (ts, level, msg) in enumerate(log_entries):
        ly = y + 0.15 + i * 0.28
        text(ax, 0.8, ly, ts, size=6, color="#7d8590", va="top", family="monospace")
        text(ax, 2.0, ly, level, size=6, color="#58a6ff", va="top", weight="bold", family="monospace")
        text(ax, 3.2, ly, msg, size=6, color="#e6edf3", va="top", family="monospace")
    
    save(fig, "09_pcc")

# ─── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating GUI3 mockups...")
    for theme in ("light", "dark"):
        print(f"\n  [{theme}]")
        set_theme(theme)
        mock_global_layout()
        mock_input_scan()
        mock_parameter()
        mock_ai_empfehlung()
        mock_run_monitor()
        mock_run_history()
        mock_raw_stack()
        mock_astrometry()
        mock_pcc()
    print(f"\nDone! PNGs saved to {os.path.join(OUT, 'mockups/')}")
