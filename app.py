# app.py — Render 可部署稳定版（HR 图全面升级：更细致刻度、区域标注、图例、PDF 同步）
from flask import Flask, render_template, request, send_file, redirect, url_for, Response

# ========== Matplotlib 字体设置 ==========
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import base64
from io import BytesIO, StringIO
import csv
from datetime import datetime
import os

# PDF / 图像工具
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# Pillow 用于处理 PNG（PDF 嵌入）
from PIL import Image

# ========== Flask app ==========
app = Flask(__name__)

# ========== 字体加载（支持 ttf / otf） ==========
FONT_PATH_TTF = "static/fonts/NotoSansSC-Regular.ttf"
FONT_PATH_OTF = "static/fonts/NotoSansSC-Regular.otf"
FONT_PATH = None
if os.path.exists(FONT_PATH_TTF):
    FONT_PATH = FONT_PATH_TTF
elif os.path.exists(FONT_PATH_OTF):
    FONT_PATH = FONT_PATH_OTF

if FONT_PATH:
    try:
        fm.fontManager.addfont(FONT_PATH)
        # 使用字体家族名（Matplotlib 会使用已注册的字体文件）
        plt.rcParams["font.family"] = "Noto Sans SC"
        plt.rcParams["axes.unicode_minus"] = False
        print("✓ 字体加载成功：", FONT_PATH)
    except Exception as e:
        print("⚠ 字体注册失败：", e)
else:
    print("⚠ 未找到字体文件，中文可能无法显示")

# ========== 国际化 ==========
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

# ========== 科学函数 ==========
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
    if L_ms <= 0:
        return "Unknown", L_ms
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

# 小工具：把英文分类映射到中文（仅用于图例 / 文本）
CATEGORY_CN = {
    "Main sequence": "主序带/主序星",
    "Giant": "巨星",
    "Hypergiant": "超巨星",
    "White dwarf": "白矮星",
    "Subsequence": "亚主序/低光度星",
    "Unknown": "未知"
}

# ========== H–R 图升级版 ==========
def plot_hr_multi(temps, lums, categories, lang='zh', max_point_labels=100):
    """
    升级版 H–R 图：
    - 更详细的刻度（横轴温度，纵轴光度）
    - 区域着色：白矮星区 / 主序带 / 巨星 / 超巨星
    - 图例（中文/英文）
    - 限制逐点文本标注数量以避免 OOM（> max_point_labels 会禁用）
    """
    # 清洗数据（保底）
    clean_t = [t if (t is not None and t > 0) else T_SUN for t in temps]
    clean_l = [l if (l is not None and l > 0) else 1.0 for l in lums]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0f0f1a", dpi=120)
    ax.set_facecolor("#0f0f1a")

    # 主序参考线（高分辨率）
    tgrid = np.logspace(np.log10(2500), np.log10(40000), 800)
    L_ms = 1e-4 * (tgrid ** 3.5)
    ax.plot(tgrid, L_ms, color="#80cfff", linewidth=2, label=("主序参考线" if lang.startswith('zh') else "Main Sequence"))

    # ========== 区域填充 ==========
    # 主序带（0.1 - 10 倍）
    ax.fill_between(tgrid, 0.1*L_ms, 10*L_ms, color="#3c9aff", alpha=0.14, linewidth=0)

    # 白矮星区（高温低光度）
    wd_mask = tgrid > 6000
    ax.fill_between(tgrid[wd_mask], 1e-6, 0.03*L_ms[wd_mask], color="#cda8ff", alpha=0.18, linewidth=0)

    # 红巨星 / 巨星区（高光度、低温也会覆盖）
    ax.fill_between(tgrid, 20*L_ms, 1e9, color="#ffbf88", alpha=0.12, linewidth=0)

    # 超巨星区（更加亮）
    ax.fill_between(tgrid, 200*L_ms, 1e10, color="#ff9b6b", alpha=0.10, linewidth=0)

    # ========== 绘制点（按分类着色） ==========
    color_map = {
        "Main sequence": "#4ea3ff", "主序星": "#4ea3ff",
        "Giant": "#ff6b6b", "巨星": "#ff6b6b",
        "Hypergiant": "#ff9b27", "超巨星": "#ff9b27",
        "White dwarf": "#cda8ff", "白矮星": "#cda8ff",
        "Subsequence": "#d0d0d0", "Unknown": "gray"
    }

    # accumulate for legend handles
    legend_handles = {}

    for i, (tt, ll, cat) in enumerate(zip(clean_t, clean_l, categories)):
        c = color_map.get(cat, "#ffffff")
        s = 60
        ax.scatter(tt, ll, s=s, color=c, edgecolors="white", linewidths=0.6, zorder=5)
        # record handle for legend
        if cat not in legend_handles:
            legend_handles[cat] = mpatches.Patch(color=c, label=(CATEGORY_CN.get(cat, cat) if lang.startswith('zh') else cat))

    # ========== 可选：逐点坐标标签（受数量限制，避免 OOM） ==========
    try:
        if len(clean_t) <= max_point_labels:
            for tt, ll in zip(clean_t, clean_l):
                ax.text(tt*1.03, ll*1.03, f"T={int(tt)}K\nL={ll:.3g}",
                        fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.38, pad=2), zorder=6)
        else:
            # 当点过多时，在右上角放一个提示文字（中文/英文）
            hint = ("点过多，已禁用逐点标注" if lang.startswith('zh') else "Too many points — per-point labels disabled")
            ax.text(0.02, 0.98, hint, transform=ax.transAxes, fontsize=9, color="white", va='top', ha='left',
                    bbox=dict(facecolor='black', alpha=0.4, pad=4))
    except Exception:
        pass

    # ========== 区域文本标签（中文/英文） ==========
    if lang.startswith('zh'):
        ax.text(3000, 6e-4, "白矮星区", fontsize=11, color="#e8d7ff")
        ax.text(3500, 8e2, "巨星区", fontsize=12, color="#6b3a00")
        ax.text(6500, 2e3, "主序带", fontsize=12, color="#cfe9ff")
        ax.text(10000, 5e4, "超巨星区", fontsize=11, color="#ffdfd0")
    else:
        ax.text(3000, 6e-4, "White Dwarf", fontsize=11, color="#e8d7ff")
        ax.text(3500, 8e2, "Giant Region", fontsize=12, color="#6b3a00")
        ax.text(6500, 2e3, "Main Sequence", fontsize=12, color="#cfe9ff")
        ax.text(10000, 5e4, "Supergiant", fontsize=11, color="#ffdfd0")

    # ========== 坐标轴：对数坐标 + 详细刻度 ==========
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 温度范围（从左到右递减）
    ax.set_xlim(40000, 2500)
    ax.set_ylim(1e-4, 1e6)

    # 温度（横轴）刻度（选定常用值）
    xticks = [40000,20000,15000,10000,9000,8000,7000,6000,5500,5000,4500,4000,3500,3000,2500]
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis='x', colors='white', rotation=45)
    # 禁止科学记数法显示（改为普通整数）
    ax.get_xaxis().get_major_formatter().set_scientific(False)

    # 光度刻度（纵轴）
    yticks = [1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6]
    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.tick_params(axis='y', colors='white')

    # 坐标轴标签与标题（中文/英文）
    ax.set_xlabel("温度 (K)" if lang.startswith('zh') else "Temperature (K)", fontsize=13, color="white")
    ax.set_ylabel("光度 (L☉)" if lang.startswith('zh') else "Luminosity (L☉)", fontsize=13, color="white")
    ax.set_title("赫罗图（H–R 图）" if lang.startswith('zh') else "H–R Diagram", fontsize=16, color="white")

    # 网格与风格
    ax.grid(True, which="both", ls=":", alpha=0.22)
    ax.tick_params(colors="white")

    # 图例（将我们记录到的分类显示）
    handles = list(legend_handles.values())
    if handles:
        legend_title = "分类" if lang.startswith('zh') else "Class"
        ax.legend(handles=handles, title=legend_title, facecolor="#202020", edgecolor="white", labelcolor="white", fontsize=9)

    # 输出为 base64 PNG
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ========== 单星内联绘图（与多星一致风格，保持精简） ==========
def plot_single_inline(temp, lum, classification, lang='zh'):
    T = temp if (temp is not None and temp > 0) else T_SUN
    L = lum if (lum is not None and lum > 0) else 1.0

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=110, facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    tgrid = np.logspace(np.log10(2500), np.log10(40000), 240)
    L_ms = 1e-4 * (tgrid ** 3.5)
    ax.plot(tgrid, L_ms, color="#80cfff", linewidth=1.2)

    ax.scatter([T], [L], c="red", s=90, edgecolors="white", linewidth=1.0, zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2500)
    ax.set_ylim(1e-4, 1e6)

    ax.set_xlabel("温度 (K)" if lang.startswith('zh') else "Temperature (K)", color="white")
    ax.set_ylabel("光度 (L☉)" if lang.startswith('zh') else "Luminosity (L☉)", color="white")
    ax.set_title("赫罗图（H–R 图）" if lang.startswith('zh') else "H–R Diagram", color="white")

    ax.grid(True, ls=":", alpha=0.2)
    ax.tick_params(colors="white")

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ========== CSV 解析 ==========
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

# ========== 生成 PDF（reportlab） — 使用 plot_hr_multi 生成高质量 H–R 图 ==========
def generate_pdf(results, temps, lums, cats, filename, lang='zh'):
    # 注册中文字体供 reportlab 使用
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_path_ttf = "static/fonts/NotoSansSC-Regular.ttf"
    font_name = "CJK"

    if os.path.exists(font_path_ttf):
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path_ttf))
        except Exception as e:
            print("PDF font registration error:", e)
            font_name = "Helvetica"
    else:
        font_name = "Helvetica"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # 用中文字体覆盖样式中的字体
    for k in styles.byName:
        try:
            styles[k].fontName = font_name
        except Exception:
            pass

    story = []

    story.append(Paragraph("恒星光谱与 H–R 图分析报告" if lang.startswith('zh') else "Stellar Spectra & H–R Report", styles['Title']))
    story.append(Paragraph(f"源文件: {filename}", styles['Normal']))
    story.append(Paragraph(f"生成时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", styles['Normal']))
    story.append(PageBreak())

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
        ('FONTNAME', (0,0), (-1,-1), font_name),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    story.append(Paragraph("分类统计" if lang.startswith('zh') else "Classification Stats", styles['Heading2']))
    story.append(Paragraph(f"总恒星数: {len(results)}" if lang.startswith('zh') else f"Total stars: {len(results)}", styles['Normal']))
    story.append(PageBreak())

    # 使用升级版 plot_hr_multi 生成 H–R 图并嵌入 PDF
    hr_img_b64 = plot_hr_multi(temps, lums, cats, lang=lang)
    hr_bytes = base64.b64decode(hr_img_b64)
    story.append(Paragraph("H–R 图" if lang.startswith('zh') else "H–R Diagram", styles['Heading2']))
    story.append(RLImage(BytesIO(hr_bytes), width=15*cm, height=11*cm))
    story.append(PageBreak())

    story.append(Paragraph("每颗恒星详细分析" if lang.startswith('zh') else "Per-star Details", styles['Heading2']))
    for r in results:
        story.append(Paragraph(f"编号: {r['index']}", styles['Heading3']))
        story.append(Paragraph(f"光谱: {r['spectral'] or '—'}", styles['Normal']))

        if r["temperature"] is not None:
            story.append(Paragraph(f"温度 (K): {r['temperature']:.1f}", styles['Normal']))
        else:
            story.append(Paragraph("温度 (K): —", styles['Normal']))

        if r["luminosity"] is not None:
            story.append(Paragraph(f"光度 (L☉): {r['luminosity']:.4g}", styles['Normal']))
        else:
            story.append(Paragraph("光度 (L☉): —", styles['Normal']))

        story.append(Paragraph(f"分类: {r['professional']}", styles['Normal']))
        R = estimate_radius(r['luminosity'], r['temperature'])
        M = estimate_mass(r['luminosity'])
        story.append(Paragraph(f"估算半径 (R☉): {R:.3f}" if R is not None else "估算半径: —", styles['Normal']))
        story.append(Paragraph(f"估算质量 (M☉): {M:.3f}" if M is not None else "估算质量: —", styles['Normal']))
        story.append(Spacer(1,8))

    doc.build(story)
    buf.seek(0)
    return buf

# ========== 已移除 GIF/动画 功能（按用户要求） ==========

# ========== Flask 路由（index / single / csv / generate_pdf / download_desktop） ==========
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
   - Python: flask, matplotlib, numpy, reportlab, pillow
   - 前端: DataTables/AOS/GSAP 可使用 CDN

如需示例 Electron skeleton 或 PyInstaller 脚本，我可生成示例并打包下载。
"""
    return Response(text, mimetype='text/plain', headers={"Content-Disposition":"attachment;filename=desktop_packaging_instructions.txt"})

# ========== 主函数 ==========
if __name__ == '__main__':
    if not FONT_PATH:
        print("⚠ Warning: static/fonts/NotoSansSC-Regular.ttf/.otf not found — Chinese labels may fallback.")
    app.run(debug=True, host='0.0.0.0', port=5000)
