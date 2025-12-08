# app.py — 全功能版：单星 + 批量 CSV + PDF + H-R 图 + 半径/质量估算 + 演化动画 + 中英双语 + 桌面说明下载
from flask import Flask, render_template, request, send_file, redirect, url_for, Response
import matplotlib
matplotlib.use('Agg')  # 无界面环境
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import base64
from io import BytesIO, StringIO
import csv
from datetime import datetime
import tempfile
import os

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# GIF writer (matplotlib uses PillowWriter internally)
from matplotlib import animation
from PIL import Image

# ---------- Matplotlib 中文支持 ----------
rcParams['font.sans-serif'] = ['SimHei']  # 保证中文显示
rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

# -------------------------
# 国际化小字典（中/英）
# 使用：t('key') 根据 lang 参数选择
# -------------------------
I18N = {
    'zh': {
        'title': '恒星光谱与 H–R 分析',
        'single': '单颗恒星分析',
        'csv': '批量 CSV 分析',
        'temperature': '温度 (K)',
        'luminosity': '光度 (L☉)',
        'spectrum': '光谱类型',
        'class': '分类',
        'radius': '估算半径 (R☉)',
        'mass': '估算质量 (M☉)',
        'generate_anim': '生成演化动画 (GIF)',
        'download_desktop': '下载桌面打包说明',
        'hr_title': '赫罗图（H–R 图）'
    },
    'en': {
        'title': 'Stellar Spectra & H–R Analysis',
        'single': 'Single Star Analysis',
        'csv': 'Batch CSV Analysis',
        'temperature': 'Temperature (K)',
        'luminosity': 'Luminosity (L☉)',
        'spectrum': 'Spectral Type',
        'class': 'Classification',
        'radius': 'Estimated Radius (R☉)',
        'mass': 'Estimated Mass (M☉)',
        'generate_anim': 'Generate Evolution GIF',
        'download_desktop': 'Download Desktop Packaging Guide',
        'hr_title': 'H–R Diagram'
    }
}

def t(key, lang='zh'):
    return I18N.get(lang, I18N['zh']).get(key, key)

# -------------------------
# 天文/物理辅助常数
# -------------------------
T_SUN = 5772.0  # K
SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W/m2/K4 — not directly used (we use normalized formula)
# L☉ and R☉ units are not required for normalized relations because we use dimensionless ratios

# -------------------------
# 基本函数（光谱→温度，分类等）
# -------------------------
def spectral_to_temperature(spectral_type):
    if not spectral_type:
        return None
    s = spectral_type.upper().strip()
    base_temp = {"O":35000,"B":15000,"A":9000,"F":7000,"G":5500,"K":4500,"M":3500}
    main = s[0]
    if main not in base_temp:
        return None
    if len(s) == 1 or not s[1].isdigit():
        return base_temp[main]
    digit = int(s[1])
    T1 = base_temp[main]
    next_class = chr(ord(main)+1)
    T2 = base_temp.get(next_class, T1 - 1000)
    return T1 - (digit/10)*(T1 - T2)

def simple_classification(temp, lum):
    if temp is None or lum is None:
        return "Unknown"
    if lum < 0.1:
        return "White dwarf"
    if lum > 10000:
        return "Hypergiant"
    if lum > 100:
        return "Giant"
    return "Main sequence"

def professional_classification(temp, lum):
    if temp is None or lum is None:
        return "Unknown", None
    L_ms = 1e-4 * (temp ** 3.5)
    if lum <= 0:
        return "Unknown", L_ms
    ratio = lum / L_ms
    if 0.1 <= ratio <= 10:
        return "Main sequence", L_ms
    if ratio > 10:
        if ratio > 1000:
            return "Hypergiant", L_ms
        return "Giant", L_ms
    if ratio < 0.05 and temp >= 6000:
        return "White dwarf", L_ms
    return "Low-luminosity/Subsequence", L_ms

# -------------------------
# 半径估算（以 R☉ 为单位）
# L/L☉ = (R/R☉)^2 * (T/T☉)^4  => R/R☉ = sqrt( (L/L☉) / (T/T☉)^4 )
# -------------------------
def estimate_radius(lum, temp):
    try:
        if lum is None or temp is None:
            return None
        ratio = (T_SUN / temp) ** 4
        R = np.sqrt(lum * ratio)
        return float(R)  # in R_sun
    except Exception:
        return None

# -------------------------
# 质量估算（以 M☉ 单位）——经验分段近似
# 使用常见经验关系（近似）：
# L ≈ 0.23 M^2.3 (M < 0.43)
# L ≈ M^4       (0.43 <= M < 2)
# L ≈ 1.5 M^3.5 (2 <= M < 20)
# L ≈ 32000 M   (M >= 20)  (粗略)
# 反求 M
# -------------------------
def estimate_mass(lum):
    try:
        if lum is None or lum <= 0:
            return None
        # try ranges by inverting approximations
        # For small L, M might be very small
        # We'll attempt piecewise inversion with boundaries by mass guess
        # Use iterative approach: try different formula ranges and pick plausible M
        # 1) assume M < 0.43
        M1 = (lum / 0.23) ** (1 / 2.3)
        if M1 < 0.43:
            return float(M1)
        # 2) assume 0.43 <= M < 2
        M2 = lum ** (1 / 4.0)
        if 0.43 <= M2 < 2.0:
            return float(M2)
        # 3) assume 2 <= M < 20
        M3 = (lum / 1.5) ** (1 / 3.5)
        if 2.0 <= M3 < 20.0:
            return float(M3)
        # 4) high mass approx
        M4 = lum / 32000.0
        if M4 >= 20.0:
            return float(M4)
        # fallback: return M2
        return float(M2)
    except Exception:
        return None

# -------------------------
# H–R 绘图函数（专业版、带中文/英文标签由前端模板控制）
# -------------------------
def plot_hr_multi(temps, lums, categories, lang='zh'):
    # prepare figure
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    tgrid = np.logspace(np.log10(2500), np.log10(40000), 500)
    L_ms = 1e-4 * (tgrid ** 3.5)

    # fill regions with labels based on lang
    ms_label = "主序带" if lang.startswith('zh') else "Main sequence band"
    giant_label = "巨星区" if lang.startswith('zh') else "Giant region"
    wd_label = "白矮星区" if lang.startswith('zh') else "White dwarf region"
    ref_label = "主序参考线" if lang.startswith('zh') else "Main sequence reference"

    ax.fill_between(tgrid, 0.1*L_ms, 10*L_ms, color="#4ea3ff", alpha=0.12, label=ms_label)
    ax.fill_between(tgrid, 10*L_ms, 1e8, color="#ffdd77", alpha=0.12, label=giant_label)
    mask = tgrid > 5000
    ax.fill_between(tgrid[mask], 1e-8, 0.05*L_ms[mask], color="#cda8ff", alpha=0.12, label=wd_label)
    ax.plot(tgrid, L_ms, color="#80cfff", linewidth=2, label=ref_label)

    # color map
    color_map = {
        "Main sequence": "#4ea3ff",
        "主序星": "#4ea3ff",
        "Giant": "#ff6b6b",
        "巨星": "#ff6b6b",
        "Hypergiant": "#ff9b27",
        "超巨星": "#ff9b27",
        "White dwarf": "#cda8ff",
        "白矮星": "#cda8ff",
        "Low-luminosity/Subsequence": "#d0d0d0",
        "亚主序星/低光度星": "#d0d0d0",
        "Unknown": "gray"
    }

    for temp, lum, cat in zip(temps, lums, categories):
        c = color_map.get(cat, "white")
        ax.scatter(temp, lum, s=60, color=c, edgecolors="white", linewidths=0.7, zorder=5)
        # label text depends on lang: categories might be English or Chinese already
        text_label = cat
        ax.text(temp*1.05, lum*1.05, text_label, fontsize=9, color=c,
                bbox=dict(facecolor="black", alpha=0.45, pad=2))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40000, 2500)
    ax.set_ylim(1e-4, 1e6)
    xlabel = "温度 (K)" if lang.startswith('zh') else "Temperature (K)"
    ylabel = "光度 (L☉)" if lang.startswith('zh') else "Luminosity (L☉)"
    ax.set_xlabel(xlabel, fontsize=13, color="white")
    ax.set_ylabel(ylabel, fontsize=13, color="white")
    title = "赫罗图（H–R Diagram）" if lang.startswith('zh') else "H–R Diagram"
    ax.set_title(title, fontsize=16, color="white")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.tick_params(colors="white")

    legend = ax.legend(facecolor="#202020", edgecolor="white", fontsize=9)
    for t in legend.get_texts():
        t.set_color("white")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# -------------------------
# 单星 H-R 图（inline，返回 base64）
# -------------------------
def plot_single_inline(temp, lum, classification, lang='zh'):
    fig, ax = plt.subplots(figsize=(7,6), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    tgrid = np.logspace(np.log10(2500), np.log10(40000), 400)
    L_ms = 1e-4 * (tgrid ** 3.5)
    ax.plot(tgrid, L_ms, color="#80cfff", linewidth=2)
    ax.scatter([temp], [lum], c="red", s=140, edgecolors="white", linewidth=1.2)
    label_text = classification
    ax.text(temp*1.1, lum*1.1, f"{label_text}\nT={int(temp)}K\nL={lum}", fontsize=11, color="white",
            bbox=dict(facecolor="black", alpha=0.5, pad=4))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(40000,2500); ax.set_ylim(1e-4,1e6)
    ax.set_xlabel("温度 (K)" if lang.startswith('zh') else "Temperature (K)", color="white")
    ax.set_ylabel("光度 (L☉)" if lang.startswith('zh') else "Luminosity (L☉)", color="white")
    ax.grid(True, ls=":", alpha=0.3); ax.tick_params(colors="white")
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# -------------------------
# CSV 解析与处理
# -------------------------
def process_csv_text(csv_text):
    reader = csv.DictReader(StringIO(csv_text))
    rows = list(reader)
    results = []
    temps, lums, cats = [], [], []
    stats = {"total": 0}
    for i, row in enumerate(rows, start=1):
        spectral = (row.get("spectral") or row.get("Spectrum") or "").strip()
        temp_s = (row.get("temperature") or row.get("Temperature") or "").strip()
        lum_s = (row.get("luminosity") or row.get("Luminosity") or "").strip()
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
        prof, Lms = professional_classification(temp, lum)
        stats["total"] += 1
        temps.append(temp)
        lums.append(lum)
        cats.append(prof)
        results.append({
            "index": i,
            "spectral": spectral,
            "temperature": temp,
            "luminosity": lum,
            "professional": prof
        })
    return results, temps, lums, cats, stats

# -------------------------
# PDF 生成
# -------------------------
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
    table_data = [["编号","光谱","温度(K)","光度(L☉)","分类" if lang.startswith('zh') else "Class"]]
    for r in results:
        table_data.append([
            r["index"],
            r["spectral"] or "—",
            f"{r['temperature']:.1f}" if r['temperature'] is not None else "—",
            f"{r['luminosity']:.4g}" if r['luminosity'] is not None else "—",
            r["professional"]
        ])
    table = Table(table_data, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#3e8cff")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.3,colors.gray),
        ('FONTSIZE',(0,0),(-1,-1),9),
    ]))
    story.append(table)
    story.append(PageBreak())
    # Stats
    story.append(Paragraph("分类统计" if lang.startswith('zh') else "Classification Stats", styles['Heading2']))
    # simple stats not maintained here; display total
    story.append(Paragraph(f"总恒星数: {len(results)}" if lang.startswith('zh') else f"Total stars: {len(results)}", styles['Normal']))
    story.append(PageBreak())
    # HR image
    story.append(Paragraph("H–R 图" if lang.startswith('zh') else "H–R Diagram", styles['Heading2']))
    hr_img = plot_hr_multi(temps, lums, cats, lang= 'zh' if lang.startswith('zh') else 'en')
    hr_data = base64.b64decode(hr_img)
    story.append(RLImage(BytesIO(hr_data), width=15*cm, height=11*cm))
    story.append(PageBreak())
    # Per-star details
    story.append(Paragraph("每颗恒星详细分析" if lang.startswith('zh') else "Per-star Details", styles['Heading2']))
    for r in results:
        story.append(Paragraph(f"编号: {r['index']}", styles['Heading3']))
        story.append(Paragraph(f"光谱: {r['spectral'] or '—'}", styles['Normal']))
        story.append(Paragraph(f"温度 (K): {(f'{r['temperature']:.1f}' if r['temperature'] is not None else '—')}", styles['Normal']))
        story.append(Paragraph(f"光度 (L☉): {(f'{r['luminosity']:.4g}' if r['luminosity'] is not None else '—')}", styles['Normal']))
        story.append(Paragraph(f"分类: {r['professional']}", styles['Normal']))
        # derived values
        R = estimate_radius(r['luminosity'], r['temperature'])
        M = estimate_mass(r['luminosity'])
        story.append(Paragraph(f"估算半径 (R☉): {R:.3f}" if R is not None else "估算半径: —", styles['Normal']))
        story.append(Paragraph(f"估算质量 (M☉): {M:.3f}" if M is not None else "估算质量: —", styles['Normal']))
        story.append(Spacer(1,8))
    doc.build(story)
    buf.seek(0)
    return buf

# -------------------------
# 演化动画（非常简化的示意轨迹）
# 采用：生成一条从主序位置向上（变亮/变冷）再收缩（白矮）之轨迹
# 输入：mass (M☉), init_temp(K), init_lum(L☉)
# 返回：GIF bytes
# -------------------------
def generate_evolution_gif(mass, init_temp, init_lum, lang='zh'):
    # Build a mock evolutionary track depending on mass
    # For low mass: slight move along MS then to WD (cooler, much less luminous)
    # For intermediate mass: move to giant (colder, brighter), then WD
    # For very high mass: move to hypergiant and then collapse (we simulate)
    steps = 60
    temps = []
    lums = []

    if mass is None:
        mass = estimate_mass(init_lum) or 1.0

    m = mass
    T0 = init_temp if init_temp else 5772
    L0 = init_lum if init_lum else 1.0

    for i in range(steps):
        frac = i / (steps - 1)
        if m < 1.0:
            # low-mass: small temp change, luminosity drops toward WD
            T = T0 * (1 - 0.2 * frac)
            L = L0 * (1 - 0.8 * frac)
            if frac > 0.7:
                T = T0 * (0.5 + 0.5*(1 - frac))  # settle cooler
                L = max(1e-4, L0 * (0.05*(1 - (frac-0.7)/0.3)))
        elif m < 8.0:
            # typical: goes to giant (cooler but brighter), then declines to WD
            if frac < 0.5:
                T = T0 * (1 - 0.4 * (frac / 0.5))
                L = L0 * (1 + 200 * (frac / 0.5))
            else:
                ff = (frac - 0.5)/0.5
                T = T0 * (0.6 + 0.4*(1 - ff))
                L = L0 * (1 + 200 * (1 - ff)) * 0.1
        else:
            # very massive: hotter and much brighter then collapse
            if frac < 0.5:
                T = T0 * (1 + 0.6*(frac/0.5))
                L = L0 * (1 + 1e4 * (frac/0.5))
            else:
                ff = (frac-0.5)/0.5
                T = T0 * (1 + 0.6*(1 - ff))
                L = max(1e-4, L0 * (1e4 * (1 - ff)))
        temps.append(max(1000, T))
        lums.append(max(1e-8, L))

    # Now draw frames and save to temporary GIF file
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmpf:
        tmpname = tmpf.name

    fig, ax = plt.subplots(figsize=(6,5), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    tgrid = np.logspace(np.log10(2500), np.log10(40000), 400)
    Lms = 1e-4 * (tgrid ** 3.5)
    ax.plot(tgrid, Lms, color='#80cfff', linewidth=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(40000,2500); ax.set_ylim(1e-6,1e6)
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel("Luminosity (L☉)")
    scat = ax.scatter([], [], c='red', s=60, edgecolors='white')

    frames = []
    for (T,L) in zip(temps, lums):
        # update scatter
        scat.set_offsets([[T,L]])
        # draw annotation
        # capture canvas to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        w,h = fig.canvas.get_width_height()
        image = image.reshape((h, w, 3))
        frames.append(Image.fromarray(image))

    # save frames to gif
    frames[0].save(tmpname, save_all=True, append_images=frames[1:], duration=80, loop=0)
    plt.close(fig)
    # read back
    with open(tmpname, 'rb') as f:
        data = f.read()
    os.remove(tmpname)
    return data

# -------------------------
# 路由：主页 / 简单导航
# -------------------------
@app.route('/')
def index():
    lang = request.args.get('lang', 'zh')
    return render_template('index.html', t=lambda k: t(k, lang), lang=lang)

# -------------------------
# 路由：单星分析（含半径/质量/动画）
# -------------------------
@app.route('/single', methods=['GET','POST'])
def single():
    lang = request.args.get('lang', 'zh')
    hr_img = None
    result = None
    anim_bytes = None

    if request.method == 'POST':
        spectral = request.form.get('spectral','').strip()
        temp_in = request.form.get('temperature','').strip()
        lum_in = request.form.get('luminosity','').strip()
        do_anim = request.form.get('do_anim','') == '1'

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
        hr_img = plot_single_inline(temp if temp else 5772, lum if lum else 1.0, prof, lang=lang)
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
            # generate gif bytes
            mass_for_anim = M if M is not None else (estimate_mass(lum) or 1.0)
            anim_bytes = generate_evolution_gif(mass_for_anim, temp if temp else 5772, lum if lum else 1.0, lang=lang)
            # return GIF directly for download
            return send_file(BytesIO(anim_bytes), mimetype='image/gif', as_attachment=True,
                             download_name=f"evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif")

    return render_template('single.html', t=lambda k: t(k, lang), lang=lang, hr_image=hr_img, result=result)

# -------------------------
# 路由：CSV 上传与预览（自动生成 hr_image）
# -------------------------
@app.route('/csv', methods=['GET','POST'])
def csv_page():
    lang = request.args.get('lang','zh')
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
        return render_template('csv_preview.html', preview_table=results, stats=stats, csv_b64=csv_b64, original_name=file.filename, hr_image=hr_image, t=lambda k: t(k, lang), lang=lang)
    return render_template('csv.html', t=lambda k: t(k, lang), lang=lang)

# -------------------------
# 路由：生成 PDF（来自预览页）
# -------------------------
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf_route():
    lang = request.args.get('lang','zh')
    csv_b64 = request.form.get('csv_b64')
    original_name = request.form.get('original_name','analysis.csv')
    if not csv_b64:
        return redirect(url_for('csv_page'))
    csv_text = base64.b64decode(csv_b64).decode('utf-8-sig')
    results, temps, lums, cats, stats = process_csv_text(csv_text)
    pdf_buf = generate_pdf(results, temps, lums, cats, original_name, lang=lang)
    out_name = f"HR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(pdf_buf, as_attachment=True, download_name=out_name, mimetype='application/pdf')

# -------------------------
# 路由：下载桌面打包说明（文本）
# -------------------------
@app.route('/download_desktop')
def download_desktop():
    text = """桌面打包说明 (简要)
1) Python + PyInstaller (将 Flask app 打包为可执行文件)
   - 安装: pip install pyinstaller
   - 命令: pyinstaller --onefile --add-data "templates:templates" --add-data "static:static" app.py
   - 说明: Windows 下 --add-data 的分隔符与路径写法不同，请参考 PyInstaller 文档。

2) 使用 Electron 打包 Web 前端为桌面应用（更推荐）
   - 用 Electron 建立一个小壳，里面用 BrowserWindow 加载本地或远程站点。
   - 可将 Flask 打包为本地可执行，再由 Electron 调用并在本地打开 http://127.0.0.1:xxxx
   - Electron 打包: 使用 electron-builder 或 electron-packager。

3) 推荐流程:
   - 在服务器上部署 (Render / Heroku / VPS)，保持服务长期在线 → 使用 Electron 仅作客户端壳。
   - 或将 Flask 与前端一起本地打包：先用 PyInstaller 打包 Flask 程序（带 templates/static），再用 Electron 载入本地 127.0.0.1。

4) 依赖:
   - Python: flask, matplotlib, numpy, reportlab, pillow
   - 前端: 无需额外，若使用 DataTables 则需 CDN 或本地静态文件。

若需要我为你生成示例 Electron 项目 skeleton 或 PyInstaller 打包脚本，我可以直接生成并打包示例文件供下载。
"""
    return Response(text, mimetype='text/plain', headers={"Content-Disposition":"attachment;filename=desktop_packaging_instructions.txt"})

# -------------------------
# 主函数（开发时使用）
# -------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
