from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import base64
from io import BytesIO
import numpy as np

# ========= 中文字体支持 =============
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# ========= Flask 应用初始化 ==========
app = Flask(__name__)


# ========== 光谱类型 → 温度转换函数 ==========
def spectral_to_temperature(spectral_type):
    """
    将光谱分类（如 G2V、A0、K5III）转换为估计温度（有效温度）
    支持：
    - 下拉框：O / B / A / F / G / K / M
    - 文本输入：如 A0, G2V, K5III
    """

    if not spectral_type:
        return None

    spectral_type = spectral_type.upper().strip()

    # 基础温度表
    base_temp = {
        "O": 35000,
        "B": 15000,
        "A": 9000,
        "F": 7000,
        "G": 5500,
        "K": 4500,
        "M": 3500
    }

    main_class = spectral_type[0]

    if main_class not in base_temp:
        return None

    # 仅主类（无数字）
    if len(spectral_type) == 1 or not spectral_type[1].isdigit():
        return base_temp[main_class]

    # 主类 + 数字（如 G2）
    digit = int(spectral_type[1])

    # 当前主类温度
    T1 = base_temp[main_class]

    # 下一光谱主类（用于插值）
    next_class = chr(ord(main_class) + 1)
    if next_class in base_temp:
        T2 = base_temp[next_class]
    else:
        T2 = T1 - 1000

    # 插值：如 G0 ~ G9
    T = T1 - (digit / 10) * (T1 - T2)
    return T


# ========== HR 图绘制函数 ==========
def plot_hr_diagram(temp, luminosity):
    """
    绘制星体在 H-R 图上的位置
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    # 反转温度轴（H-R 图惯例）
    ax.set_xlim(40000, 2500)

    # 绘制主序星区域背景
    ax.scatter(temp, luminosity, color='red', s=80, label="目标恒星")

    ax.set_xlabel("温度 (K)")
    ax.set_ylabel("光度 (L / L☉)")
    ax.set_title("赫罗图（H–R Diagram）")

    ax.legend()

    # 输出为 base64
    pngImage = BytesIO()
    fig.savefig(pngImage, format="png", bbox_inches='tight')
    pngImage.seek(0)
    encoded = base64.b64encode(pngImage.getvalue()).decode('utf-8')
    plt.close(fig)
    return encoded


# ========== 首页路由 ==========
@app.route("/", methods=["GET", "POST"])
def index():
    hr_image = None

    if request.method == "POST":

        # 获取表单内容
        spectral_dropdown = request.form.get("spectral_dropdown", "")
        spectral_text     = request.form.get("spectral_text", "")
        temp_input        = request.form.get("temperature", "")
        luminosity_input  = request.form.get("luminosity", "")

        # ----------- 处理温度（优先级：手动输入 > 文本光谱 > 下拉光谱） -----------

        if temp_input.strip() != "":
            temperature = float(temp_input)

        else:
            # 尝试文本输入光谱
            if spectral_text.strip() != "":
                temperature = spectral_to_temperature(spectral_text)
            # 尝试下拉框
            elif spectral_dropdown.strip() != "":
                temperature = spectral_to_temperature(spectral_dropdown)
            else:
                temperature = None

        if temperature is None:
            return render_template("index.html", hr_image=None)

        # 光度
        luminosity = float(luminosity_input)

        # ----------- 绘制 HR 图 -----------
        hr_image = plot_hr_diagram(temperature, luminosity)

    return render_template("index.html", hr_image=hr_image)


# ========== 启动应用（Render 使用 gunicorn，不走这里） ==========
if __name__ == "__main__":
    app.run(debug=True)
