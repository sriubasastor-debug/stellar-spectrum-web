# app.py — Render 可部署稳定版

from flask import Flask, render_template, request, send_file, redirect, url_for, Response

# ========== Matplotlib 字体设置 ==========
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import base64
from io import BytesIO, StringIO
import csv
from datetime import datetime
import tempfile
import os

from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm


# ============================================================
# 1) 先创建 Flask app（必须在最前面，否则会 NameError）
# ============================================================

app = Flask(__name__)


# ============================================================
# 2) 字体加载（支持 ttf / otf）
# ============================================================

FONT_PATH_TTF = "static/fonts/NotoSansSC-Regular.ttf"
FONT_PATH_OTF = "static/fonts/NotoSansSC-Regular.otf"

FONT_PATH = None

if os.path.exists(FONT_PATH_TTF):
    FONT_PATH = FONT_PATH_TTF
elif os.path.exists(FONT_PATH_OTF):
    FONT_PATH = FONT_PATH_OTF

if FONT_PATH:
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["axes.unicode_minus"] = False
    print("✓ 字体加载成功：", FONT_PATH)
else:
    print("⚠ 未找到字体文件，中文可能无法显示")


# ============================================================
# 3) 国际化
# ============================================================

I18N = {
    "zh": {
        "title": "恒星光谱与 H–R 分析",
        "single": "单颗恒星分析",
        "csv": "批量 CSV 分析",
        "temperature": "温度 (K)",
        "luminosity": "光度 (L☉)",
        "spectrum": "光谱类型",
        "class": "分类",
        "radius": "估算半径 (R☉)",
        "mass": "估算质量 (M☉)",
        "generate_anim": "生成演化动画 (GIF)",
        "download_desktop": "下载桌面打包说明",
        "hr_title": "赫罗图（H–R 图）"
    },
    "en": {
        "title": "Stellar Spectra & H–R Analysis",
        "single": "Single Star Analysis",
        "csv": "Batch CSV Analysis",
        "temperature": "Temperature (K)",
        "luminosity": "Luminosity (L☉)",
        "spectrum": "Spectral Type",
        "class": "Classification",
        "radius": "Estimated Radius (R☉)",
        "mass": "Estimated Mass (M☉)",
        "generate_anim": "Generate Evolution GIF",
        "download_desktop": "Download Desktop Packaging Guide",
        "hr_title": "H–R Diagram"
    }
}

def t(key, lang="zh"):
    return I18N.get(lang, I18N["zh"]).get(key, key)


# ============================================================
# 4) 科学函数
# ============================================================

T_SUN = 5772


def spectral_to_temperature(s):
    if not s:
        return None
    s = s.upper().strip()
    table = {"O":35000,"B":15000,"A":9000,"F":7000,"G":5500,"K":4500,"M":3500}
    if s[0] not in table:
        return None
    T0 = table[s[0]]
    if len(s) >= 2 and s[1].isdigit():
        n = int(s[1])
        next_class = chr(ord(s[0])+1)
        T1 = table.get(next_class, T0-1000)
        return T0 - (n/10)*(T0-T1)
    return T0


def professional_classification(temp, lum):
    if temp is None or lum is None:
        return "Unknown", None
    L_ms = 1e-4 * temp**3.5
    r = lum / L_ms
    if 0.1 <= r <= 10:
        return "Main sequence", L_ms
    if r > 10:
        if r > 1000:
            return "Hypergiant", L_ms
        return "Giant", L_ms
    if r < 0.05 and temp >= 6000:
        return "White dwarf", L_ms
    return "Subsequence", L_ms


def estimate_radius(lum, temp):
    if lum is None or temp is None or temp <= 0:
        return None
    return float((lum * (T_SUN/temp)**4)**0.5)


def estimate_mass(lum):
    if lum is None or lum <= 0:
        return None
    if lum < 0.23*(0.43**2.3):
        return float((lum/0.23)**(1/2.3))
    M2 = lum**0.25
    if M2 < 2:
        return float(M2)
    M3 = (lum/1.5)**(1/3.5)
    if 2 <= M3 < 20:
        return float(M3)
    return float(lum/32000)
# ============================================================
# H–R 图（多点，返回 base64 PNG）
# ============================================================
def plot_hr_multi(temps, lums, categories, lang='zh'):
    # 保证绘图数据有效
    clean_t = [t if (t is not None and t > 0) else T_SUN for t in temps]
    clean_l = [l if (l is not None and l > 0) else 1.0 for l in lums]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    tgrid = np.logspace(np.log10(2500), np.log10(40000), 500)
    L_ms = 1e-4 * (tgrid ** 3.5)

    # region fills & reference line
    ax.fill_between(tgrid, 0.1 * L_ms, 10 * L_ms, color="#4ea3ff", alpha=0.12)
    ax.fill_between(tgrid, 10 * L_ms, 1e8, color="#ffdd77", alpha=0.12)
    mask = tgrid > 5000
    ax.fill_between(tgrid[mask], 1e-8, 0.05 * L_ms[mask], color="#cda8ff", alpha=0.12)
    ax.plot(tgrid, L_ms, color="#80cfff", linewidth=2)

    color_map = {
        "Main sequence": "#4ea3ff", "主序星": "#4ea3ff",
        "Giant": "#ff6b6b", "巨星": "#ff6b6b",
        "Hypergiant": "#ff9b27", "超巨星": "#ff9b27",
        "White dwarf": "#cda8ff", "白矮星": "#cda8ff",
        "Subsequence": "#d0d0d0", "Unknown": "gray"
    }

    for t, l, cat in zip(clean_t, clean_l, categories):
        c = color_map.get(cat, "#ffffff")
        ax.scatter(t, l, s=50, color=c, edgecolors="white", linewidths=0.6, zorder=5)
        try:
            ax.text(t * 1.03, l * 1.03, str(cat), fontsize=8, color=c,
                    bbox=dict(facecolor="black", alpha=0.45, pad=2))
        except Exception:
            pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2500)
    ax.set_ylim(1e-4, 1e6)

    xlabel = "温度 (K)" if lang.startswith('zh') else "Temperature (K)"
    ylabel = "光度 (L☉)" if lang.startswith('zh') else "Luminosity (L☉)"
    ax.set_xlabel(xlabel, fontsize=13, color="white")
    ax.set_ylabel(ylabel, fontsize=13, color="white")
    ax.set_title("赫罗图（H–R 图）" if lang.startswith('zh') else "H–R Diagram", fontsize=16, color="white")
    ax.grid(True, which="both", ls=":", alpha=0.25)
    ax.tick_params(colors="white", which='both')

    legend = ax.legend(facecolor="#202020", edgecolor="white", fontsize=9)
    for ttext in legend.get_texts():
        ttext.set_color("white")

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ============================================================
# 单星内联 H–R 图（返回 base64）
# ============================================================
def plot_single_inline(temp, lum, classification, lang='zh'):
    T = temp if (temp is not None and temp > 0) else T_SUN
    L = lum if (lum is not None and lum > 0) else 1.0

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    tgrid = np.logspace(np.log10(2500), np.log10(40000), 400)
    L_ms = 1e-4 * (tgrid ** 3.5)
    ax.plot(tgrid, L_ms, color="#80cfff", linewidth=2)
    ax.scatter([T], [L], c="red", s=140, edgecolors="white", linewidth=1.2, zorder=6)
    try:
        ax.text(T * 1.05, L * 1.05, f"{classification}\nT={int(T)}K\nL={L}", fontsize=10, color="white",
                bbox=dict(facecolor="black", alpha=0.5, pad=4))
    except Exception:
        pass
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2500)
    ax.set_ylim(1e-4, 1e6)
    ax.set_xlabel("温度 (K)" if lang.startswith('zh') else "Temperature (K)", color="white")
    ax.set_ylabel("光度 (L☉)" if lang.startswith('zh') else "Luminosity (L☉)", color="white")
    ax.grid(True, ls=":", alpha=0.25)
    ax.tick_params(colors="white")
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ============================================================
# CSV 解析与处理
# ============================================================
def process_csv_text(csv_text):
    reader = csv.DictReader(StringIO(csv_text))
    rows = list(reader)
    results = []
    temps, lums, cats = [], [], []
    stats = {"total": 0}
    for i, row in enumerate(rows, start=1):
        spectral = (row.get("spectral") or row.get("Spectrum") or "").strip()
        temp_s = (row.get("temperature") or row.get("Temperature") or row.get("temp") or "").strip()
        lum_s = (row.get("luminosity") or row.get("Luminosity") or row.get("lum") or "").strip()

        temp = None
        if temp_s:
            try:
                temp = float(temp_s)
            except:
                temp = None
        else:
            temp = spectral_to_temperature(spectral)

        try:
            lum = float(lum_s) if lum_s else None
        except:
            lum = None

        prof, _ = professional_classification(temp, lum)
        temps.append(temp)
        lums.append(lum)
        cats.append(prof)

        results.append({
            "index": i,
            "spectral": spectral or "",
            "temperature": temp,
            "luminosity": lum,
            "professional": prof
        })
        stats["total"] += 1

    return results, temps, lums, cats, stats


# ============================================================
# 生成 PDF（使用 reportlab）
# ============================================================
def generate_pdf(results, temps, lums, cats, filename, lang='zh'):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Cover
    story.append(Paragraph("恒星光谱与 H–R 图分析报告" if lang.startswith('zh') else "Stellar Spectra & H–R Report", styles['Title']))
    story.append(Paragraph(f"源文件: {filename}", styles['Normal']))
    story.append(Paragraph(f"生成时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", styles['Normal']))
    story.append(PageBreak())

    # Table
    story.append(Paragraph("数据表格" if lang.startswith('zh') else "Data Table", styles['Heading2']))
    header = ["编号", "光谱", "温度(K)", "光度(L☉)", "分类"] if lang.startswith('zh') else ["#", "Spectrum", "Temperature(K)", "Luminosity(L☉)", "Class"]
    table_data = [header]
    for r in results:
        table_data.append([
            r["index"],
            r["spectral"] or "—",
            f"{r['temperature']:.1f}" if r['temperature'] is not None else "—",
            f"{r['luminosity']:.4g}" if r['luminosity'] is not None else "—",
            r["professional"]
        ])
    tbl = Table(table_data, repeatRows=1, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#3e8cff")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.3, colors.gray),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # Stats
    story.append(Paragraph("分类统计" if lang.startswith('zh') else "Classification Stats", styles['Heading2']))
    story.append(Paragraph(f"总恒星数: {len(results)}" if lang.startswith('zh') else f"Total stars: {len(results)}", styles['Normal']))
    story.append(PageBreak())

    # HR image
    story.append(Paragraph("H–R 图" if lang.startswith('zh') else "H–R Diagram", styles['Heading2']))
    hr_img_b64 = plot_hr_multi(temps, lums, cats, lang=lang)
    hr_bytes = base64.b64decode(hr_img_b64)
    story.append(RLImage(BytesIO(hr_bytes), width=15*cm, height=11*cm))
    story.append(PageBreak())

    # Per-star details
    story.append(Paragraph("每颗恒星详细分析" if lang.startswith('zh') else "Per-star Details", styles['Heading2']))
    for r in results:
        story.append(Paragraph(f"编号: {r['index']}", styles['Heading3']))
        story.append(Paragraph(f"光谱: {r['spectral'] or '—'}", styles['Normal']))
        story.append(Paragraph(f"温度 (K): {(f'{r['temperature']:.1f}' if r['temperature'] is not None else '—')}", styles['Normal']))
        story.append(Paragraph(f"光度 (L☉): {(f'{r['luminosity']:.4g}' if r['luminosity'] is not None else '—')}", styles['Normal']))
        story.append(Paragraph(f"分类: {r['professional']}", styles['Normal']))
        R = estimate_radius(r['luminosity'], r['temperature'])
        M = estimate_mass(r['luminosity'])
        story.append(Paragraph(f"估算半径 (R☉): {R:.3f}" if R is not None else "估算半径: —", styles['Normal']))
        story.append(Paragraph(f"估算质量 (M☉): {M:.3f}" if M is not None else "估算质量: —", styles['Normal']))
        story.append(Spacer(1,8))

    doc.build(story)
    buf.seek(0)
    return buf


# ============================================================
# 生成演化 GIF（示意轨迹）
# ============================================================
def generate_evolution_gif(mass, init_temp, init_lum, lang='zh'):
    steps = 60
    temps = []
    lums = []

    if mass is None:
        mass = estimate_mass(init_lum) or 1.0

    m = mass
    T0 = init_temp if init_temp else T_SUN
    L0 = init_lum if init_lum else 1.0

    for i in range(steps):
        frac = i / (steps - 1)
        if m < 1.0:
            T = T0 * (1 - 0.2 * frac)
            L = L0 * (1 - 0.8 * frac)
            if frac > 0.7:
                T = T0 * (0.5 + 0.5 * (1 - frac))
                L = max(1e-4, L0 * (0.05 * (1 - (frac - 0.7) / 0.3)))
        elif m < 8.0:
            if frac < 0.5:
                T = T0 * (1 - 0.4 * (frac / 0.5))
                L = L0 * (1 + 200 * (frac / 0.5))
            else:
                ff = (frac - 0.5) / 0.5
                T = T0 * (0.6 + 0.4 * (1 - ff))
                L = L0 * (1 + 200 * (1 - ff)) * 0.1
        else:
            if frac < 0.5:
                T = T0 * (1 + 0.6 * (frac / 0.5))
                L = L0 * (1 + 1e4 * (frac / 0.5))
            else:
                ff = (frac - 0.5) / 0.5
                T = T0 * (1 + 0.6 * (1 - ff))
                L = max(1e-4, L0 * (1e4 * (1 - ff)))
        temps.append(max(1000, T))
        lums.append(max(1e-8, L))

    # render frames
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    tgrid = np.logspace(np.log10(2500), np.log10(40000), 400)
    Lms = 1e-4 * (tgrid ** 3.5)
    ax.plot(tgrid, Lms, color='#80cfff', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(40000, 2500)
    ax.set_ylim(1e-6, 1e6)
    scat = ax.scatter([], [], c='red', s=60, edgecolors='white')

    frames = []
    for (T, L) in zip(temps, lums):
        scat.set_offsets([[T, L]])
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        image = image.reshape((h, w, 3))
        frames.append(Image.fromarray(image))

    # save to temp file
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmpf:
        tmpname = tmpf.name
    frames[0].save(tmpname, save_all=True, append_images=frames[1:], duration=80, loop=0)
    plt.close(fig)
    with open(tmpname, 'rb') as f:
        data = f.read()
    os.remove(tmpname)
    return data


# ============================================================
# Flask 路由（index / single / csv / generate_pdf / download_desktop）
# ============================================================

@app.route('/')
def index():
    lang = request.args.get('lang', 'zh')
    return render_template('index.html', t=lambda k: t(k, lang), lang=lang)


@app.route('/single', methods=['GET', 'POST'])
def single():
    lang = request.args.get('lang', 'zh')
    hr_img = None
    result = None

    if request.method == 'POST':
        spectral = request.form.get('spectral', '').strip()
        temp_in = request.form.get('temperature', '').strip()
        lum_in = request.form.get('luminosity', '').strip()
        do_anim = request.form.get('do_anim', '') == '1'

        temp = None
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

        prof, _ = professional_classification(temp, lum)
        hr_img = plot_single_inline(temp if temp else T_SUN, lum if lum else 1.0, prof, lang=lang)
        R = estimate_radius(lum, temp)
        M = estimate_mass(lum)

        result = {
            "spectral": spectral,
            "temperature": temp,
            "luminosity": lum,
            "professional": prof,
            "radius": R,
            "mass": M
        }

        if do_anim:
            mass_for_anim = M if M is not None else (estimate_mass(lum) or 1.0)
            anim_bytes = generate_evolution_gif(mass_for_anim, temp if temp else T_SUN, lum if lum else 1.0, lang=lang)
            return send_file(BytesIO(anim_bytes), mimetype='image/gif', as_attachment=True,
                             download_name=f"evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif")

    return render_template('single.html', t=lambda k: t(k, lang), lang=lang, hr_image=hr_img, result=result)


@app.route('/csv', methods=['GET', 'POST'])
def csv_page():
    lang = request.args.get('lang', 'zh')
    if request.method == 'POST':
        file = request.files.get('csvfile')
        if not file:
            return redirect(url_for('csv_page'))
        content = file.read()
        try:
            data = content.decode('utf-8-sig')
        except:
            data = content.decode('latin-1')

        results, temps, lums, cats, stats = process_csv_text(data)
        hr_image = plot_hr_multi(temps, lums, cats, lang=lang)
        csv_b64 = base64.b64encode(data.encode('utf-8')).decode('ascii')

        return render_template(
            'csv_preview.html',
            preview_table=results,
            stats=stats,
            csv_b64=csv_b64,
            original_name=file.filename,
            hr_image=hr_image,
            t=lambda k: t(k, lang),
            lang=lang
        )
    return render_template('csv.html', t=lambda k: t(k, lang), lang=lang)


@app.route('/generate_pdf', methods=['POST'])
def generate_pdf_route():
    lang = request.args.get('lang', 'zh')
    csv_b64 = request.form.get('csv_b64')
    original_name = request.form.get('original_name', 'analysis.csv')
    if not csv_b64:
        return redirect(url_for('csv_page'))
    try:
        csv_text = base64.b64decode(csv_b64).decode('utf-8-sig')
    except:
        csv_text = base64.b64decode(csv_b64).decode('latin-1')

    results, temps, lums, cats, stats = process_csv_text(csv_text)
    pdf_buf = generate_pdf(results, temps, lums, cats, original_name, lang=lang)
    out_name = f"HR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(pdf_buf, as_attachment=True, download_name=out_name, mimetype='application/pdf')


@app.route('/download_desktop')
def download_desktop():
    text = """桌面打包说明 (简要)
1) Python + PyInstaller (将 Flask app 打包为可执行文件)
   - 安装: pip install pyinstaller
   - 命令: pyinstaller --onefile --add-data "templates;templates" --add-data "static;static" app.py
   - Windows 下 --add-data 的分隔符是分号 ; (如上)，Linux/macOS 使用冒号 :

2) Electron 打包 Web 前端为桌面应用（推荐）
   - 用 Electron 建立壳，BrowserWindow 加载本地或远程站点。
   - 将 Flask 打包为本地可执行，再由 Electron 调用并在本地打开 http://127.0.0.1:xxxx
   - electron-builder / electron-packager 可创建安装包。

3) 推荐流程:
   - 先在服务器上部署（Render / VPS），长期在线，Electron 仅作客户端壳。
   - 或将 Flask 与前端一并本地打包：PyInstaller 打包 Flask（带 templates/static），Electron 载入。

4) 依赖:
   - Python: flask, matplotlib, numpy, reportlab, pillow, pandas
   - 前端: DataTables/AOS/GSAP 可使用 CDN

如需示例 Electron skeleton 或 PyInstaller 脚本，我可生成示例并打包下载。
"""
    return Response(text, mimetype='text/plain', headers={"Content-Disposition":"attachment;filename=desktop_packaging_instructions.txt"})


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    if not FONT_PATH:
        print("⚠ Warning: static/fonts/NotoSansSC-Regular.ttf/.otf not found — Chinese labels may fallback.")
    app.run(debug=True, host='0.0.0.0', port=5000)
