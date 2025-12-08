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

# PDF 生成库
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# ---------- Matplotlib 中文支持 ----------
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

#=============================
# 工具函数
#=============================

def spectral_to_temperature(spectral_type):
    if not spectral_type:
        return None
    s = spectral_type.upper().strip()
    base_temp = {
        "O": 35000, "B": 15000, "A": 9000,
        "F": 7000, "G": 5500, "K": 4500, "M": 3500
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


def simple_classification(temp, lum):
    if temp is None or lum is None:
        return "无法分类"
    if lum < 0.1: return "白矮星"
    if lum > 10000: return "超巨星"
    if lum > 100: return "巨星"
    return "主序星"


def professional_classification(temp, lum):
    if temp is None or lum is None:
        return "无法分类", None

    L_ms = 1e-4 * (temp ** 3.5)
    if lum <= 0:
        return "无法分类", L_ms

    ratio = lum / L_ms

    if 0.1 <= ratio <= 10:
        return "主序星", L_ms
    if ratio > 10:
        if ratio > 1000: return "超巨星", L_ms
        return "巨星", L_ms
    if ratio < 0.05 and temp >= 6000:
        return "白矮星", L_ms

    return "亚主序星/低光度星", L_ms


#=============================
# 升级版赫罗图：批量绘制
#=============================

def plot_hr_multi(temps, lums, categories):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    tgrid = np.logspace(np.log10(2500), np.log10(40000), 500)
    L_ms = 1e-4 * (tgrid ** 3.5)

    ax.fill_between(tgrid, 0.1*L_ms, 10*L_ms,
                    color="#4ea3ff", alpha=0.12, label="主序带")

    ax.fill_between(tgrid, 10*L_ms, 1e8,
                    color="#ffdd77", alpha=0.12, label="巨星区")

    mask = tgrid > 5000
    ax.fill_between(tgrid[mask], 1e-8, 0.05*L_ms[mask],
                    color="#cda8ff", alpha=0.12, label="白矮星区")

    ax.plot(tgrid, L_ms, color="#80cfff", linewidth=2, label="主序参考线")

    color_map = {
        "主序星": "#4ea3ff",
        "巨星": "#ff6b6b",
        "超巨星": "#ff9b27",
        "白矮星": "#cda8ff",
        "亚主序星/低光度星": "#d0d0d0",
        "无法分类": "gray"
    }

    for temp, lum, cat in zip(temps, lums, categories):
        c = color_map.get(cat, "white")
        ax.scatter(temp, lum, s=55, color=c, edgecolors="white", linewidths=0.7)
        ax.text(temp*1.05, lum*1.05, cat, fontsize=9, color=c,
                bbox=dict(facecolor="black", alpha=0.5, pad=2))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2500)
    ax.set_ylim(1e-4, 1e6)

    ax.set_xlabel("温度 (K)", fontsize=13, color="white")
    ax.set_ylabel("光度 (L☉)", fontsize=13, color="white")
    ax.set_title("赫罗图（H–R Diagram）", fontsize=16, color="white")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.tick_params(colors="white")

    legend = ax.legend(facecolor="#202020", edgecolor="white",
                       fontsize=10, labelcolor="white")
    for t in legend.get_texts():
        t.set_color("white")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


#=============================
# 单星专用 H-R 图（网页显示）
#=============================

def plot_single_inline(temp, lum, classification):
    fig, ax = plt.subplots(figsize=(7,6), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    tgrid = np.logspace(np.log10(2500), np.log10(40000), 500)
    L_ms = 1e-4 * (tgrid ** 3.5)

    ax.plot(tgrid, L_ms, color="#80cfff", linewidth=2)

    ax.scatter([temp], [lum], c="red", s=120, edgecolors="white", linewidth=1.2)
    ax.text(temp*1.1, lum*1.1,
            f"{classification}\nT={int(temp)}K\nL={lum}",
            fontsize=11, color="white",
            bbox=dict(facecolor="black", alpha=0.5, pad=4))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2500)
    ax.set_ylim(1e-4, 1e6)

    ax.set_xlabel("温度 (K)", color="white")
    ax.set_ylabel("光度 (L☉)", color="white")
    ax.grid(True, ls=":", alpha=0.3)
    ax.tick_params(colors="white")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


#=============================
# CSV 解析、预览和 PDF 生成
#=============================

def process_csv_text(csv_text):
    reader = csv.DictReader(StringIO(csv_text))
    rows = list(reader)

    results = []
    temps, lums, cats = [], [], []
    stats = {"total": 0}

    for i, row in enumerate(rows, start=1):
        spectral = row.get("spectral", "").strip()
        temp_s = row.get("temperature", "").strip()
        lum_s = row.get("luminosity", "").strip()

        temp = float(temp_s) if temp_s else spectral_to_temperature(spectral)
        lum = float(lum_s) if lum_s else None

        simple = simple_classification(temp, lum)
        prof, _ = professional_classification(temp, lum)

        category = prof
        stats["total"] += 1

        temps.append(temp)
        lums.append(lum)
        cats.append(category)

        results.append({
            "index": i,
            "spectral": spectral,
            "temperature": temp,
            "luminosity": lum,
            "simple": simple,
            "professional": category,
        })

    return results, temps, lums, cats, stats


def generate_pdf(results, temps, lums, cats, filename):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("恒星光谱与 H–R 图分析报告", styles['Title']))
    story.append(Paragraph(f"源文件：{filename}", styles['Normal']))
    story.append(PageBreak())

    # 数据表
    table_data = [["编号","光谱","温度","光度","分类"]]
    for r in results:
        table_data.append([
            r["index"], r["spectral"], r["temperature"],
            r["luminosity"], r["professional"]
        ])
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#4ea3ff")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.3,colors.gray),
    ]))
    story.append(table)
    story.append(PageBreak())

    # HR图
    hr_img = plot_hr_multi(temps, lums, cats)
    hr_data = base64.b64decode(hr_img)
    story.append(RLImage(BytesIO(hr_data), width=14*cm, height=10*cm))

    doc.build(story)
    buf.seek(0)
    return buf


#=============================
# 网页路由
#=============================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/single", methods=["GET","POST"])
def single():
    hr_img = None
    result = None

    if request.method == "POST":
        spectral = request.form.get("spectral","")
        temp_input = request.form.get("temperature","")
        lum_input = request.form.get("luminosity","")

        temp = float(temp_input) if temp_input else spectral_to_temperature(spectral)
        lum = float(lum_input) if lum_input else None

        cat, _ = professional_classification(temp, lum)
        hr_img = plot_single_inline(temp, lum, cat)

        result = {
            "spectral": spectral,
            "temperature": temp,
            "luminosity": lum,
            "category": cat
        }

    return render_template("single.html", hr_image=hr_img, result=result)


@app.route("/csv", methods=["GET","POST"])
def csv_page():
    if request.method == "POST":
        file = request.files.get("csvfile")
        if not file:
            return redirect("/csv")

        data = file.read().decode("utf-8-sig")
        results, temps, lums,cats, stats = process_csv_text(data)

        csv_b64 = base64.b64encode(file.read()).decode('utf-8')

        return render_template("csv_preview.html",
                               preview_table=results,
                               stats=stats,
                               csv_b64=base64.b64encode(data.encode()).decode(),
                               original_name=file.filename)

    return render_template("csv.html")


@app.route("/generate_pdf", methods=["POST"])
def generate_pdf_route():
    csv_b64 = request.form.get("csv_b64")
    filename = request.form.get("original_name","analysis.csv")

    csv_text = base64.b64decode(csv_b64).decode("utf-8-sig")

    results, temps, lums,cats, stats = process_csv_text(csv_text)
    pdf_buf = generate_pdf(results, temps, lums,cats, filename)

    pdf_name = f"HR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(pdf_buf, as_attachment=True, download_name=pdf_name,
                     mimetype="application/pdf")


if __name__ == "__main__":
    app.run(debug=True)
