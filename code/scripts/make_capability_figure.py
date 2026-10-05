#!/usr/bin/env python3
"""Render the capability-curve figure (effect vs baseline) as a self-contained SVG.
Two series (What/spec, Where/location) x four models, with 95% cluster-bootstrap CI
error bars. CVD-safe Okabe-Ito colours + distinct marker shapes + direct labels."""
import json, os

from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
curve = json.load(open(f"{ROOT}/generated/second_model/capability_curve.json"))
# order by baseline
curve = sorted(curve, key=lambda r: r["baseline_pct"])

W, H = 460, 300
ML, MR, MT, MB = 52, 92, 16, 40
PW, PH = W - ML - MR, H - MT - MB
X0, X1 = 20, 90          # baseline % range
Y0, Y1 = -10, 62         # effect pp range
BLUE, ORANGE = "#0072B2", "#E69F00"

def sx(v): return ML + (v - X0) / (X1 - X0) * PW
def sy(v): return MT + (Y1 - v) / (Y1 - Y0) * PH

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
# zero line
s.append(f'<line x1="{ML}" y1="{sy(0):.1f}" x2="{ML+PW}" y2="{sy(0):.1f}" stroke="#bbb" stroke-width="1" stroke-dasharray="3,3"/>')
# 10-pt material bar
s.append(f'<line x1="{ML}" y1="{sy(10):.1f}" x2="{ML+PW}" y2="{sy(10):.1f}" stroke="#ddd" stroke-width="1"/>')
s.append(f'<text x="{ML+PW-2}" y="{sy(10)-3:.1f}" font-size="8" fill="#999" text-anchor="end">10-pt material bar</text>')
# axes
s.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#333" stroke-width="1"/>')
s.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#333" stroke-width="1"/>')
for v in range(0, 61, 20):
    s.append(f'<text x="{ML-6}" y="{sy(v)+3:.1f}" font-size="9" fill="#555" text-anchor="end">{v}</text>')
    s.append(f'<line x1="{ML-3}" y1="{sy(v):.1f}" x2="{ML}" y2="{sy(v):.1f}" stroke="#333"/>')
for v in (25, 40, 55, 70, 85):
    s.append(f'<text x="{sx(v):.1f}" y="{MT+PH+13}" font-size="9" fill="#555" text-anchor="middle">{v}</text>')
    s.append(f'<line x1="{sx(v):.1f}" y1="{MT+PH}" x2="{sx(v):.1f}" y2="{MT+PH+3}" stroke="#333"/>')
s.append(f'<text x="{ML+PW/2:.1f}" y="{H-6}" font-size="10" fill="#333" text-anchor="middle">no-spec/no-location baseline success (%)</text>')
s.append(f'<text x="14" y="{MT+PH/2:.1f}" font-size="10" fill="#333" text-anchor="middle" transform="rotate(-90 14 {MT+PH/2:.1f})">effect on repair success (percentage points)</text>')

def series(key, ci, color, shape, label, ly):
    pts = [(r["baseline_pct"], r[key], r[ci]) for r in curve]
    # connecting line
    d = " ".join(f"{'M' if i==0 else 'L'}{sx(x):.1f} {sy(y):.1f}" for i,(x,y,_) in enumerate(pts))
    s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.55"/>')
    for x, y, c in pts:
        s.append(f'<line x1="{sx(x):.1f}" y1="{sy(c[0]):.1f}" x2="{sx(x):.1f}" y2="{sy(c[1]):.1f}" stroke="{color}" stroke-width="1.2" opacity="0.7"/>')
        s.append(f'<line x1="{sx(x)-3:.1f}" y1="{sy(c[0]):.1f}" x2="{sx(x)+3:.1f}" y2="{sy(c[0]):.1f}" stroke="{color}" stroke-width="1.2" opacity="0.7"/>')
        s.append(f'<line x1="{sx(x)-3:.1f}" y1="{sy(c[1]):.1f}" x2="{sx(x)+3:.1f}" y2="{sy(c[1]):.1f}" stroke="{color}" stroke-width="1.2" opacity="0.7"/>')
        if shape == "circle":
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{color}" stroke="white" stroke-width="1"/>')
        else:
            s.append(f'<path d="M{sx(x):.1f} {sy(y)-5:.1f} L{sx(x)+4.6:.1f} {sy(y)+3.5:.1f} L{sx(x)-4.6:.1f} {sy(y)+3.5:.1f} Z" fill="{color}" stroke="white" stroke-width="1"/>')
    # direct label at right edge
    lx, lyv = pts[-1][0], pts[-1][1]
    s.append(f'<text x="{ML+PW+6}" y="{ly:.1f}" font-size="10" fill="{color}" font-weight="bold">{label}</text>')

series("what_pp", "what_ci", BLUE, "circle", "What (spec)", sy(4.5)+3)
series("where_pp", "where_ci", ORANGE, "triangle", "Where (loc)", sy(4.5)+16)
s.append('</svg>')
open(f"{ROOT}/generated/second_model/capability_curve.svg", "w").write("\n".join(s))
print("wrote capability_curve.svg")
