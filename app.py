# app.py — Flask 网页主程序

from flask import Flask, render_template, request
from stellar_spectra.core import classify_by_temperature, calculate_physical_parameters, plot_hr_diagram
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io, base64

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        temp = float(request.form["temp"])

        spectral_type = classify_by_temperature(temp)
        if spectral_type is None:
            return render_template("index.html", error="温度超出可分类范围！")

        params = calculate_physical_parameters(spectral_type, temp)

        # 绘制 HR 图并转为 base64 图片
        fig = plot_hr_diagram(temp, params["luminosity_value"])
        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        return render_template(
            "index.html",
            spectral_type=spectral_type,
            mass=params["mass"],
            radius=params["radius"],
            luminosity=params["luminosity"],
            plot_data=plot_data
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

