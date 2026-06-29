#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the Chinese Markdown training report into a self-contained styled HTML
report with an inline SVG loss curve. Uses only the stdlib + the system `markdown`
package (no pip installs). Output: BEVFusion_NAVSIM_训练报告.html
"""
import os
import markdown  # system package

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "BEVFusion_NAVSIM_训练报告.md")
OUT = os.path.join(HERE, "BEVFusion_NAVSIM_训练报告.html")

# --------------------------------------------------------------------------- #
# Inline SVG loss curve (windowed means over the 36-epoch run).
# x = iter in thousands (0..172), y = loss value (0..18).
# --------------------------------------------------------------------------- #
SERIES = {
    "loss_total": ("#1e5aa0", [(1, 17.1), (12, 12.4), (36, 11.2), (62, 9.86), (90, 9.33),
                                (110, 9.12), (132, 8.78), (150, 8.71), (162, 8.62), (170, 8.0)]),
    "loss_bbox":  ("#be5a0a", [(1, 13.9), (12, 9.3), (36, 8.1), (62, 6.88), (90, 6.42),
                                (110, 6.23), (132, 5.92), (150, 5.80), (162, 5.75), (170, 5.4)]),
    "loss_heatmap": ("#148240", [(1, 2.93), (12, 3.0), (36, 2.79), (62, 2.75), (90, 2.70),
                                  (110, 2.68), (132, 2.64), (150, 2.70), (162, 2.65), (170, 2.6)]),
    "loss_cls":   ("#8e44ad", [(1, 0.28), (12, 0.21), (36, 0.22), (62, 0.22), (90, 0.21),
                                (110, 0.21), (132, 0.22), (150, 0.22), (162, 0.21), (170, 0.21)]),
}

W, H = 760, 380
PAD_L, PAD_R, PAD_T, PAD_B = 60, 150, 20, 45
XMAX, YMAX = 172.0, 18.0
PW = W - PAD_L - PAD_R
PH = H - PAD_T - PAD_B


def sx(x):
    return PAD_L + (x / XMAX) * PW


def sy(y):
    return PAD_T + (1 - y / YMAX) * PH


def build_svg():
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'style="max-width:100%;height:auto;font-family:sans-serif;">']
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    # grid + y labels
    for yv in range(0, int(YMAX) + 1, 2):
        yy = sy(yv)
        parts.append(f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{PAD_L+PW}" y2="{yy:.1f}" '
                     f'stroke="#e6e9ee" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{yy+3:.1f}" font-size="11" fill="#888" '
                     f'text-anchor="end">{yv}</text>')
    # x labels
    for xv in range(0, 181, 30):
        xx = sx(xv)
        parts.append(f'<text x="{xx:.1f}" y="{PAD_T+PH+18:.1f}" font-size="11" fill="#888" '
                     f'text-anchor="middle">{xv}</text>')
    parts.append(f'<text x="{PAD_L+PW/2:.1f}" y="{H-6}" font-size="12" fill="#555" '
                 f'text-anchor="middle">训练 iter（千）</text>')
    parts.append(f'<text x="16" y="{PAD_T+PH/2:.1f}" font-size="12" fill="#555" '
                 f'text-anchor="middle" transform="rotate(-90 16 {PAD_T+PH/2:.1f})">损失值</text>')
    # axes
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+PH}" stroke="#999" stroke-width="1.2"/>')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T+PH}" x2="{PAD_L+PW}" y2="{PAD_T+PH}" stroke="#999" stroke-width="1.2"/>')
    # reference dashed line: old-run bbox stall ~10
    parts.append(f'<line x1="{PAD_L}" y1="{sy(10):.1f}" x2="{PAD_L+PW}" y2="{sy(10):.1f}" '
                 f'stroke="#d44" stroke-width="1.3" stroke-dasharray="6 4"/>')
    # series
    legend_y = PAD_T + 6
    for name, (color, pts) in SERIES.items():
        d = "M " + " L ".join(f"{sx(x):.1f} {sy(y):.1f}" for x, y in pts)
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6" fill="{color}"/>')
        lx = PAD_L + PW + 14
        parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx+22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+28}" y="{legend_y+4}" font-size="12" fill="#333">{name}</text>')
        legend_y += 22
    # legend: stall reference
    lx = PAD_L + PW + 14
    parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx+22}" y2="{legend_y}" stroke="#d44" '
                 f'stroke-width="2" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{lx+28}" y="{legend_y+4}" font-size="11" fill="#333">旧训练停滞≈10</text>')
    parts.append('</svg>')
    return "\n".join(parts)


CSS = """
:root { --accent:#1e5aa0; --good:#148240; --warn:#be5a0a; }
* { box-sizing:border-box; }
body {
  font-family:"Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif;
  color:#222; line-height:1.65; max-width:920px; margin:0 auto; padding:40px 32px 80px;
  background:#fff;
}
h1 { color:var(--accent); font-size:30px; border-bottom:3px solid var(--accent);
     padding-bottom:14px; margin-top:0; }
h2 { color:var(--accent); font-size:22px; margin-top:38px;
     border-left:5px solid var(--accent); padding-left:12px; }
h3 { color:#2c5f8a; font-size:17px; margin-top:26px; }
p, li { font-size:14.5px; }
code { background:#f1f4f8; padding:1px 5px; border-radius:4px; font-size:90%;
       font-family:"DejaVu Sans Mono",monospace; color:#b1442a; }
table { border-collapse:collapse; width:100%; margin:16px 0; font-size:13.5px; }
th, td { border:1px solid #d8dde4; padding:7px 10px; text-align:left; }
th { background:var(--accent); color:#fff; font-weight:600; }
tr:nth-child(even) td { background:#f6f8fb; }
blockquote { border-left:4px solid #f0b050; background:#fff8ec; margin:14px 0;
             padding:8px 16px; color:#7a5a10; border-radius:0 6px 6px 0; }
strong { color:#1a1a1a; }
hr { border:none; border-top:1px solid #e0e4ea; margin:32px 0; }
.figure { text-align:center; margin:24px 0; padding:16px; background:#fbfcfe;
          border:1px solid #e4e8ee; border-radius:8px; }
.figcap { font-size:12.5px; color:#777; margin-top:8px; }
.cover-meta { font-size:14px; color:#444; }
ul, ol { padding-left:22px; }
"""


def main():
    with open(MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Inject the SVG figure right after the "### 4.1 训练损失轨迹" table marker.
    # We place it under section 4 heading "## 4. 训练过程与损失曲线".
    svg = build_svg()
    fig_block = (
        '\n\n<div class="figure">\n' + svg +
        '\n<div class="figcap">图 1：训练损失下降曲线（窗口均值）。'
        'loss_bbox 在 epoch ~2.5 突破旧训练停滞的 ≈10（红色虚线），随后随余弦学习率持续缓降。</div>\n</div>\n\n'
    )
    anchor = "### 4.1 训练损失轨迹（窗口均值）"
    if anchor in md_text:
        md_text = md_text.replace(anchor, fig_block + anchor, 1)

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )

    html = (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>BEVFusion-on-NAVSIM 3D 检测训练全流程报告</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{html_body}\n</body>\n</html>\n"
    )

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("written:", OUT, "(%d bytes)" % os.path.getsize(OUT))


if __name__ == "__main__":
    main()
