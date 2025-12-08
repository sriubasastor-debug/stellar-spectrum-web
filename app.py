from flask import Flask, render_template, request, send_file, redirect, url_for
import matplotlib
# 使用 SimHei 字体，支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import base64
from io import BytesIO, StringIO
import csv
from datetime import datetime

# PDF 生成
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# ========= Matplotlib 中文支持 ==========
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

# ========= 光谱 → 温度 ==========
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
    return T1 - (digit / 10) * (T1 - T2)

# ========= 简单分类 ==========
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

# ========= 专业分类（主序线对比） ==========
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
        return ("超巨星" if ratio > 1000 else "巨星", L_ms)
    if temp >= 6000 and ratio < 0.05:
        return ("白矮星 (可能)", L_ms)
    return ("低光度主序/亚星 (可能)", L_ms)

# ========= 单星 H-R 图绘制 ==========
def plot_single_star(temp, lum, classification):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a1a")

    temps = np.logspace(np.log10(2500), np.log10(40000), 400)
    L_ms = 1e-4 * (temps ** 3.5)
    ax.plot(temps, L_ms, label="主序线", color="#87CEFA")

    # 区域带
    ax.fill_between(temps, 0.1 * L_ms, 10 * L_ms, color="#6bb4ff", alpha=0.15)
    ax.fill_between(temps, 10 * L_ms, 1e8, color="#FFD580", alpha=0.12)
    wmask = temps > 5000
    ax.fill_between(temps[wmask], 1e-8, 0.05*L_ms[wmask], color="#d0b7ff", alpha=0.15)

    # 星点
    ax.scatter([temp], [lum], c="red", s=120, edgecolors="white", zorder=5)

    ax.text(
        temp * 1.05,
        lum * 1.1,
        f"{classification}\nT={int(temp)} K\nL={lum}",
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5)
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2500)
    ax.set_ylim(1e-4, 1e6)
    ax.set_xlabel("温度 (K)")
    ax.set_ylabel("光度 (L☉)")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.set_title("H–R 图（单星）", color="white")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ========= 批量 H-R 图 ==========
def plot_hr_multi(temps, lums):
    fig, ax = plt.subplots(figsize=(8,6), facecolor="#1a1a1a")

    tgrid = np.logspace(np.log10(2500), np.log10(40000), 400)
    L_ms = 1e-4 * (tgrid ** 3.5)
    ax.plot(tgrid, L_ms, color="#87CEFA")

    ax.fill_between(tgrid, 0.1*L_ms, 10*L_ms, color="#6bb4ff", alpha=0.15)
    ax.fill_between(tgrid, 10*L_ms, 1e8, color="#FFD580", alpha=0.12)
    m = tgrid > 5000
    ax.fill_between(tgrid[m], 1e-8, 0.05*L_ms[m], color="#d0b7ff", alpha=0.15)

    ax.scatter(temps, lums, c="red", s=40, edgecolors="white")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2500)
    ax.set_xlabel("温度 (K)")
    ax.set_ylabel("光度 (L☉)")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.set_title("H–R 图（批量）", color="white")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# ========= CSV 解析 ==========
def process_csv(stream):
    text = stream.read().decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))

    results = []
    temps = []
    lums = []
    stats = {"total":0, "simple":{}, "professional":{}}

    i = 0
    for row in reader:
        i += 1
        stats["total"] += 1
        spectral = row.get("spectral","").strip()
        temp_s = row.get("temperature","").strip()
        lum_s = row.get("luminosity","").strip()

        # 光度
        lum = float(lum_s) if lum_s else None

        # 温度
        if temp_s:
            try:
                temp = float(temp_s)
            except:
                temp = None
        else:
            temp = spectral_to_temperature(spectral) if spectral else None

        # 分类
        simple = simple_classification(temp, lum)
        prof, L_ms = professional_classification(temp, lum)

        stats["simple"][simple] = stats["simple"].get(simple, 0) + 1
        stats["professional"][prof] = stats["professional"].get(prof, 0) + 1

        results.append({
            "index": i,
            "spectral": spectral,
            "temperature": temp,
            "luminosity": lum,
            "simple": simple,
            "professional": prof
        })

        if temp and lum:
            temps.append(temp)
            lums.append(lum)

    return results, temps, lums, stats

# ========= PDF 生成 ==========
def generate_pdf(results, temps, lums, stats, filename):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # 封面
    story.append(Paragraph("恒星光谱与 H–R 图分析报告", styles["Title"]))
    story.append(Paragraph(f"文件：{filename}", styles["Normal"]))
    story.append(Paragraph(f"总恒星数：{stats['total']}", styles["Normal"]))
    story.append(Paragraph(
        f"生成时间：{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        styles["Normal"]
    ))
    story.append(PageBreak())

    # 目录
    story.append(Paragraph("目录", styles["Heading2"]))
    story.append(Paragraph("1. 数据表格", styles["Normal"]))
    story.append(Paragraph("2. 分类统计", styles["Normal"]))
    story.append(Paragraph("3. H–R 图", styles["Normal"]))
    story.append(Paragraph("4. 每颗恒星详细分析", styles["Normal"]))
    story.append(PageBreak())

    # 1. 数据表格
    story.append(Paragraph("1. 数据表格", styles["Heading2"]))
    table_data = [["编号","光谱","温度","光度","简单分类","专业分类"]]
    for r in results:
        table_data.append([
            r["index"],
            r["spectral"] or "—",
            f"{r['temperature']:.1f}" if r["temperature"] else "—",
            f"{r['luminosity']:.3g}" if r["luminosity"] else "—",
            r["simple"],
            r["professional"]
        ])
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#3e8cff")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.3,colors.gray)
    ]))
    story.append(table)
    story.append(PageBreak())

    # 2. 分类统计
    story.append(Paragraph("2. 分类统计", styles["Heading2"]))
    story.append(Paragraph("简单分类：", styles["Heading3"]))
    for k,v in stats["simple"].items():
        story.append(Paragraph(f"{k}：{v} 颗", styles["Normal"]))
    story.append(Spacer(1,10))

    story.append(Paragraph("专业分类：", styles["Heading3"]))
    for k,v in stats["professional"].items():
        story.append(Paragraph(f"{k}：{v} 颗", styles["Normal"]))
    story.append(PageBreak())

    # 3. HR 图
    story.append(Paragraph("3. H–R 图", styles["Heading2"]))
    imgbuf = plot_hr_multi(temps, lums)
    story.append(RLImage(imgbuf, width=15*cm, height=11*cm))
    story.append(PageBreak())

    # 4. 每颗星详细分析
    story.append(Paragraph("4. 每颗恒星详细分析", styles["Heading2"]))
    for r in results:
        story.append(Paragraph(f"编号：{r['index']}", styles["Heading3"]))
        story.append(Paragraph(f"光谱：{r['spectral'] or '—'}", styles["Normal"]))
        story.append(Paragraph(f"温度：{r['temperature']}", styles["Normal"]))
        story.append(Paragraph(f"光度：{r['luminosity']}", styles["Normal"]))
        story.append(Paragraph(f"简单分类：{r['simple']}", styles["Normal"]))
        story.append(Paragraph(f"专业分类：{r['professional']}", styles["Normal"]))
        story.append(Spacer(1,12))

    doc.build(story)
    buf.seek(0)
    return buf

# ========= 路由 ==========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/single", methods=["GET","POST"])
def single():
    hr_img = None
    result = None
    if request.method == "POST":
        spectral = request.form.get("spectral","")
        temp_in = request.form.get("temperature","")
        lum_in = request.form.get("luminosity","")

        # 处理温度
        if temp_in:
            try:
                temp = float(temp_in)
            except:
                temp = None
        else:
            temp = spectral_to_temperature(spectral)

        try:
            lum = float(lum_in) if lum_in else None
        except:
            lum = None

        if temp and lum:
            simple = simple_classification(temp, lum)
            prof, _ = professional_classification(temp, lum)
            result = f"简单分类：{simple}<br>专业分类：{prof}"
            hr_img = plot_single_star(temp, lum, prof)

    return render_template("single.html", hr_image=hr_img, result=result)

@app.route("/csv", methods=["GET","POST"])
def csv_page():
    if request.method == "POST":
        file = request.files.get("csvfile")
        if not file:
            return redirect("/csv")

        results, temps, lums, stats = process_csv(file.stream)
        pdf = generate_pdf(results, temps, lums, stats, file.filename)

        name = f"HR_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(pdf, as_attachment=True, download_name=name)

    return render_template("csv.html")

if __name__ == "__main__":
    app.run(debug=True)

