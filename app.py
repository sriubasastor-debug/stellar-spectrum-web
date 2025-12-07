# app.py — 支持 CSV 上传并生成带目录与每颗恒星详细分析的 PDF 报告
from flask import Flask, render_template, request, send_file, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import base64
from io import BytesIO, StringIO
import csv
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# ========== Matplotlib 中文配置 ==========
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# ========== Flask ==========
app = Flask(__name__)

# ========== 光谱 → 温度 ==========
def spectral_to_temperature(spectral_type):
    if not spectral_type:
        return None
    s = spectral_type.upper().strip()
    base_temp = {
        "O": 35000,
        "B": 15000,
        "A": 9000,
        "F": 7000,
        "G": 5500,
        "K": 4500,
        "M": 3500
    }
    main = s[0]
    if main not in base_temp:
        return None
    if len(s) == 1 or not s[1].isdigit():
        return base_temp[main]
    digit = int(s[1])
    T1 = base_temp[main]
    next_class = chr(ord(main) + 1)
    T2 = base_temp.get(next_class, T1 - 1000)
    T = T1 - (digit / 10) * (T1 - T2)
    return T

# ========== 简单分类 ==========
def simple_classification(temp, lum):
    if temp is None or lum is None:
        return "无法分类"
    if lum < 0.1:
        return "白矮星 (可能)"
    if lum > 10000:
        return "超巨星 (可能)"
    if lum > 100:
        return "巨星 (可能)"
    return "主序星 (可能)"

# ========== 专业分类 ==========
def professional_classification(temp, lum):
    if temp is None or lum is None:
        return ("无法分类", None)
    L_ms = 1e-4 * (temp ** 3.5)
    if lum <= 0:
        return ("无法分类", L_ms)
    ratio = lum / L_ms
    if 0.1 <= ratio <= 10:
        return ("主序星 (与主序线一致)", L_ms)
    if ratio > 10:
        if ratio > 1000:
            return ("超巨星 (远高于主序线)", L_ms)
        return ("巨星/亮巨星 (高于主序线)", L_ms)
    if temp >= 6000 and ratio < 0.05:
        return ("白矮星 (可能)", L_ms)
    return ("低光度主序/亚星 (可能)", L_ms)

# ========== 绘图（支持批量点）==========
def plot_hr_multi(temps, lums, show_regions=True):
    """
    绘制批量 H-R 图，返回 PNG bytes
    temps, lums: 数组或列表
    """
    fig, ax = plt.subplots(figsize=(8,6), facecolor='#1a1a1a')
    # 主序曲线
    temps_grid = np.logspace(np.log10(2500), np.log10(40000), 400)
    L_ms_curve = 1e-4 * (temps_grid ** 3.5)
    ax.plot(temps_grid, L_ms_curve, label="主序参考线", color="#87CEFA", linewidth=1.2)

    if show_regions:
        ax.fill_between(temps_grid, 0.1*L_ms_curve, 10*L_ms_curve, color="#89cff0", alpha=0.12, label="主序带")
        ax.fill_between(temps_grid, 10*L_ms_curve, 1e8, color="#FFD580", alpha=0.10, label="巨星/超巨星区域")
        wd_mask_temp = temps_grid > 5000
        wd_upper = 0.05 * L_ms_curve
        ax.fill_between(temps_grid[wd_mask_temp], 1e-8, wd_upper[wd_mask_temp], color="#D1C4E9", alpha=0.12, label="白矮星示意区")

    # 绘制点
    ax.scatter(temps, lums, c="red", edgecolors='white', linewidths=0.6, s=40, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2500)
    ymin = max(1e-4, min(min(lums)*0.1 if lums else 1e-4, 1e-4))
    ymax = max(max(lums)*10 if lums else 1e2, 1e6)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel("温度 (K)")
    ax.set_ylabel("光度 (L / L☉)")
    ax.set_title("H–R 图（批量）")
    ax.grid(True, which="both", ls=":", color="gray", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

# ========== CSV 解析与结果生成 ==========
def process_csv_file(file_stream, filename_hint="uploaded.csv"):
    """
    接受文件流（text mode），解析 CSV（支持有 header 或无 header）
    返回：results_list（每行 dict）、temps_list、lums_list、stats dict
    """
    # 读取文件文本并解析为 DictReader
    text = file_stream.read().decode('utf-8-sig')
    reader = csv.DictReader(StringIO(text))
    # If no header or missing expected columns, try fallback by index
    rows = list(reader)
    # If CSV had no headers, DictReader will treat first row as header; handle minimal case:
    if not rows and text.strip() != "":
        # try csv.reader fallback
        reader2 = csv.reader(StringIO(text))
        rows2 = list(reader2)
        # Expect columns: spectral, temperature, luminosity
        results = []
        for r in rows2:
            # pad to length 3
            r = r + ['']*(3-len(r))
            results.append({"spectral": r[0], "temperature": r[1], "luminosity": r[2]})
        rows = results

    results_out = []
    temps = []
    lums = []
    stats = {"total":0, "simple":{}, "professional":{}}

    idx = 0
    for r in rows:
        idx += 1
        stats["total"] += 1
        spectral = (r.get("spectral") or r.get("Spectral") or r.get("spectrum") or "").strip()
        temp_str = (r.get("temperature") or r.get("Temperature") or "").strip()
        lum_str = (r.get("luminosity") or r.get("luminosity(L)") or r.get("Luminosity") or "").strip()

        # parse luminosity
        try:
            lum = float(lum_str) if lum_str != "" else None
        except:
            lum = None

        # temperature priority
        temp_val = None
        if temp_str != "":
            try:
                temp_val = float(temp_str)
            except:
                temp_val = None
        elif spectral != "":
            temp_val = spectral_to_temperature(spectral)
        else:
            temp_val = None

        simple_cls = simple_classification(temp_val, lum)
        prof_cls, L_ms = professional_classification(temp_val, lum)

        # collect stats
        stats["simple"][simple_cls] = stats["simple"].get(simple_cls, 0) + 1
        stats["professional"][prof_cls] = stats["professional"].get(prof_cls, 0) + 1

        # record
        results_out.append({
            "index": idx,
            "spectral": spectral,
            "temperature": temp_val,
            "luminosity": lum,
            "simple_class": simple_cls,
            "professional_class": prof_cls,
            "L_ms": L_ms
        })

        if temp_val is not None and lum is not None:
            temps.append(temp_val)
            lums.append(lum)

    return results_out, temps, lums, stats

# ========== PDF 生成 ==========
def generate_pdf_report(results, temps, lums, stats, original_filename="uploaded.csv"):
    """
    使用 ReportLab 生成 PDF，返回 BytesIO
    包含：封面、目录、数据表、分类统计、HR 图、每颗恒星详细分析
    """
    buffer = BytesIO()
    # Use landscape A4 for wide tables
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Cover
    title = "恒星光谱与 H–R 图分析报告"
    story.append(Paragraph(title, styles['Title']))
    meta = f"生成时间：{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)<br/>源星表：{original_filename}<br/>总恒星数：{stats.get('total',0)}"
    story.append(Paragraph(meta, styles['Normal']))
    story.append(Spacer(1,12))

    # Table of Contents (simple clickable TOC is complex; we provide manual section list)
    story.append(Paragraph("目录", styles['Heading2']))
    toc_items = ["1. 数据表格", "2. 分类统计", "3. H–R 图", "4. 每颗恒星详细分析"]
    for it in toc_items:
        story.append(Paragraph(it, styles['Normal']))
    story.append(PageBreak())

    # 1. Data table
    story.append(Paragraph("1. 数据表格", styles['Heading2']))
    table_data = [["编号", "光谱", "温度 (K)", "光度 (L☉)", "简单分类", "专业分类"]]
    for r in results:
        temp_str = f"{r['temperature']:.1f}" if r['temperature'] is not None else "—"
        lum_str = f"{r['luminosity']:.4g}" if r['luminosity'] is not None else "—"
        table_data.append([r['index'], r['spectral'] or "—", temp_str, lum_str, r['simple_class'], r['professional_class']])
    tbl = Table(table_data, repeatRows=1, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#3e8cff")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('GRID',(0,0),(-1,-1),0.3,colors.gray),
        ('FONTSIZE',(0,0),(-1,-1),9),
    ]))
    story.append(tbl)
    story.append(Spacer(1,12))

    # 2. Classification stats
    story.append(Paragraph("2. 分类统计", styles['Heading2']))
    # simple stats
    story.append(Paragraph("简单分类统计：", styles['Heading3']))
    for k,v in stats.get('simple',{}).items():
        story.append(Paragraph(f"{k}：{v} 颗", styles['Normal']))
    story.append(Spacer(1,8))
    story.append(Paragraph("专业分类统计：", styles['Heading3']))
    for k,v in stats.get('professional',{}).items():
        story.append(Paragraph(f"{k}：{v} 颗", styles['Normal']))
    story.append(PageBreak())

    # 3. HR 图
    story.append(Paragraph("3. H–R 图", styles['Heading2']))
    # generate HR image
    img_buf = plot_hr_multi(temps, lums)
    # embed image
    rl_img = RLImage(img_buf, width=16*cm, height=12*cm)  # scale to fit
    story.append(rl_img)
    story.append(PageBreak())

    # 4. Per-star detail
    story.append(Paragraph("4. 每颗恒星的详细分析", styles['Heading2']))
    for r in results:
        lines = []
        lines.append(Paragraph(f"编号：{r['index']}", styles['Heading3']))
        lines.append(Paragraph(f"光谱：{r['spectral'] or '—'}", styles['Normal']))
        lines.append(Paragraph(f"温度 (K)：{(f'{r['temperature']:.1f}' if r['temperature'] is not None else '—')}", styles['Normal']))
        lines.append(Paragraph(f"光度 (L☉)：{(f'{r['luminosity']:.4g}' if r['luminosity'] is not None else '—')}", styles['Normal']))
        lines.append(Paragraph(f"简单分类：{r['simple_class']}", styles['Normal']))
        lines.append(Paragraph(f"专业分类：{r['professional_class']}", styles['Normal']))
        if r.get('L_ms') is not None:
            lines.append(Paragraph(f"与主序线光度比值：{(r['luminosity']/r['L_ms'] if (r['luminosity'] and r['L_ms']) else '—')}", styles['Normal']))
        story.extend(lines)
        story.append(Spacer(1,8))
    # done
    doc.build(story)
    buffer.seek(0)
    return buffer

# ========== Flask 路由 ==========
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", hr_image=None, message=None)

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """
    接受 CSV 上传，生成 PDF，并返回 PDF 文件供下载
    """
    uploaded = request.files.get("csvfile")
    if not uploaded:
        return redirect(url_for('index'))

    # parse CSV
    results, temps, lums, stats = process_csv_file(uploaded.stream, filename_hint=uploaded.filename)

    # generate PDF
    pdf_buffer = generate_pdf_report(results, temps, lums, stats, original_filename=uploaded.filename)
    # prepare filename
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_name = f"analysis_report_{ts}.pdf"

    # send file
    return send_file(pdf_buffer, as_attachment=True, download_name=out_name, mimetype='application/pdf')

# ========== 运行 ==========
if __name__ == "__main__":
    app.run(debug=True)
