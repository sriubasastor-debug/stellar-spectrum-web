from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import base64
from io import BytesIO
import numpy as np

# ========= 中文字体支持 ============
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# ========= Flask 应用 ============
app = Flask(__name__)

# ========= 光谱 → 温度（保留先前逻辑） ============
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
    # next spectral class
    next_class = chr(ord(main) + 1)
    T2 = base_temp.get(next_class, T1 - 1000)
    T = T1 - (digit / 10) * (T1 - T2)
    return T

# ========= 简单分类（按温度） ============
def simple_classification(temp, lum):
    """
    简单粗略分类（基于温度/光度阈值），返回字符串
    """
    if temp is None or lum is None:
        return "无法分类"
    # 简单规则（启发式）
    if lum < 0.1:
        return "白矮星 (可能)"
    if lum > 10000:
        return "超巨星 (可能)"
    if lum > 100:
        return "巨星 (可能)"
    # 其余视为主序星（近似）
    return "主序星 (可能)"

# ========= 专业分类（基于 H-R 位置：与主序线比较） ============
def professional_classification(temp, lum):
    """
    更专业的判定：通过比较目标光度与主序线 (L_ms ~ 1e-4 * T^3.5)
    - 如果 lum 在 [0.1*L_ms, 10*L_ms] -> 主序星
    - 如果 lum > 10*L_ms -> 巨星/超巨星（按倍数）
    - 如果 lum < 0.1*L_ms -> 可能为白矮星或次主序（按温度判别）
    返回 (分类字符串, L_ms)
    """
    if temp is None or lum is None:
        return ("无法分类", None)

    # 主序线经验模型（示意）
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
    # ratio < 0.1
    # 低于主序线很多 -> 可能是白矮星（如果温度偏高）或褐矮/低光度主序
    if temp >= 6000 and ratio < 0.05:
        return ("白矮星 (可能)", L_ms)
    return ("低光度主序/亚星 (可能)", L_ms)

# ========= 绘图（在图上标注区域与点） ============
def plot_hr_diagram_with_regions(temp, luminosity, classification_text=None, L_ms=None):
    # 建立图像（温度横轴为对数并通常从高到低）
    fig, ax = plt.subplots(figsize=(7, 6))

    # 生成温度数组（用于主序线）
    temps = np.logspace(np.log10(2500), np.log10(40000), 400)
    L_ms_curve = 1e-4 * (temps ** 3.5)

    # 绘制主序线
    ax.plot(temps, L_ms_curve, label="主序参考线 (示意)", color="#87CEFA", linewidth=1.5)

    # 绘制主序带（0.1*L_ms 到 10*L_ms）
    ax.fill_between(temps, 0.1 * L_ms_curve, 10 * L_ms_curve, color="#89cff0", alpha=0.12, label="主序带")

    # 绘制巨星区域（高于 10*L_ms）
    ax.fill_between(temps, 10 * L_ms_curve, 1e8, color="#FFD580", alpha=0.10, label="巨星/超巨星区域")

    # 绘制白矮星示意区（低光度、高温区）: 定义为 lum < 0.05*L_ms 且 temp > 5000
    wd_mask_temp = temps > 5000
    wd_upper = 0.05 * L_ms_curve
    ax.fill_between(temps[wd_mask_temp], 1e-8, wd_upper[wd_mask_temp], color="#D1C4E9", alpha=0.12, label="白矮星示意区")

    # 绘制目标点
    ax.scatter([temp], [luminosity], c="red", s=100, edgecolors='white', linewidths=0.8, zorder=5, label="目标恒星")

    # 标注目标点的文本（放在点右上）
    txt = f"T={temp:.0f} K\nL={luminosity:.3g} L☉"
    if classification_text:
        txt += f"\n{classification_text}"
    ax.text(temp * 1.05, luminosity * 1.05, txt, fontsize=10, color="white",
            bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.3'))

    # 设置为对数-对数图（惯例）
    ax.set_xscale("log")
    ax.set_yscale("log")

    # H-R 图通常将温度从高到低显示（右高左低），因此反转 x 轴
    ax.set_xlim(40000, 2500)
    # y 轴范围自动
    ax.set_ylim(bottom=max(1e-4, min(luminosity / 1000, 1e-4)), top=max(luminosity * 100, 1e6))

    ax.set_xlabel("温度 (K)")
    ax.set_ylabel("光度 (L / L☉)")
    ax.set_title("赫罗图（H–R Diagram） — 含示意区域")
    ax.grid(True, which="both", ls=":", color="gray", alpha=0.3)

    ax.legend(loc="upper left", fontsize=9)

    # 输出为 base64
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150, facecolor='#1a1a1a')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_b64

# ========= 路由 ============
@app.route("/", methods=["GET", "POST"])
def index():
    hr_image = None
    simple_class = None
    professional_class = None
    used_temp = None
    L_ms = None

    if request.method == "POST":
        spectral_dropdown = request.form.get("spectral_dropdown", "")
        spectral_text     = request.form.get("spectral_text", "")
        temp_input        = request.form.get("temperature", "")
        luminosity_input  = request.form.get("luminosity", "")

        # 处理温度（优先级：手动输入 > 文本光谱 > 下拉光谱）
        if temp_input and temp_input.strip() != "":
            try:
                used_temp = float(temp_input)
            except:
                used_temp = None
        else:
            if spectral_text and spectral_text.strip() != "":
                used_temp = spectral_to_temperature(spectral_text)
            elif spectral_dropdown and spectral_dropdown.strip() != "":
                used_temp = spectral_to_temperature(spectral_dropdown)
            else:
                used_temp = None

        # 处理光度
        try:
            luminosity = float(luminosity_input)
        except:
            luminosity = None

        # 如果缺参数则返回空页面（模板会提示）
        if used_temp is None or luminosity is None:
            return render_template("index.html", hr_image=None, error="请提供有效的温度及光度（或至少用光谱推温）")

        # 简单分类
        simple_class = simple_classification(used_temp, luminosity)

        # 专业分类
        professional_class, L_ms = professional_classification(used_temp, luminosity)

        # 绘图并标注
        hr_image = plot_hr_diagram_with_regions(used_temp, luminosity, classification_text=professional_class, L_ms=L_ms)

    return render_template("index.html",
                           hr_image=hr_image,
                           simple_class=simple_class,
                           professional_class=professional_class,
                           used_temp=used_temp)
    
if __name__ == "__main__":
    app.run(debug=True)
