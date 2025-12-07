# core.py — 恒星光谱分类 + HR 图绘制核心逻辑

from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

import numpy as np
from matplotlib.figure import Figure
from .data import (
    TEMPERATURE_RANGES,
    MASS_RANGES,
    RADIUS_RANGES,
    LUMINOSITY_RANGES
)

def classify_by_temperature(temp):
    """根据温度判断光谱类型"""
    for sp, (low, high) in TEMPERATURE_RANGES.items():
        if low <= temp < high:
            return sp
    return None


def calculate_physical_parameters(spectral_type, temp):
    """根据光谱类型计算物理参数（平均值）"""
    mass_low, mass_high = MASS_RANGES[spectral_type]
    radius_low, radius_high = RADIUS_RANGES[spectral_type]
    lum_low, lum_high = LUMINOSITY_RANGES[spectral_type]

    return {
        "mass": f"{(mass_low + mass_high) / 2:.2f} 个太阳质量",
        "radius": f"{(radius_low + radius_high) / 2:.2f} 个太阳半径",
        "luminosity": f"{(lum_low + lum_high) / 2:.2f} 倍太阳光度",

        "luminosity_value": (lum_low + lum_high) / 2,
    }


def plot_hr_diagram(temp, luminosity):
    """绘制 HR 图并返回 Figure 对象"""
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    temps = np.logspace(3, 5, 300)
    lums = 1e-4 * temps ** 3.5
    ax.plot(temps, lums, label="主序星", color="blue")

    ax.scatter([temp], [luminosity], color="red", s=80, label="目标恒星")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("温度 (K)")
    ax.set_ylabel("光度 (太阳 = 1)")
    ax.set_title("H–R 赫罗图")
    ax.legend()
    ax.grid(True)

    return fig
