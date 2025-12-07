# data.py — 恒星光谱常量数据

SPECTRAL_TYPES = {
    "O": {"color": "Blue", "peak_temp": 35000},
    "B": {"color": "Blue-White", "peak_temp": 15000},
    "A": {"color": "White", "peak_temp": 9000},
    "F": {"color": "Yellow-White", "peak_temp": 6500},
    "G": {"color": "Yellow", "peak_temp": 5500},
    "K": {"color": "Orange", "peak_temp": 4500},
    "M": {"color": "Red", "peak_temp": 3000},
}

TEMPERATURE_RANGES = {
    "M": (2000, 3500),
    "K": (3500, 5000),
    "G": (5000, 6000),
    "F": (6000, 7500),
    "A": (7500, 10000),
    "B": (10000, 30000),
    "O": (30000, float("inf")),
}

MASS_RANGES = {
    "O": (15, 90),
    "B": (2.1, 16),
    "A": (1.4, 2.1),
    "F": (1.04, 1.4),
    "G": (0.8, 1.04),
    "K": (0.45, 0.8),
    "M": (0.08, 0.45),
}

RADIUS_RANGES = {
    "O": (6.6, 15),
    "B": (1.8, 6.6),
    "A": (1.4, 1.8),
    "F": (1.15, 1.4),
    "G": (0.96, 1.15),
    "K": (0.7, 0.96),
    "M": (0.2, 0.7),
}

LUMINOSITY_RANGES = {
    "O": (30000, 1000000),
    "B": (25, 30000),
    "A": (5, 25),
    "F": (1.5, 5),
    "G": (0.6, 1.5),
    "K": (0.08, 0.6),
    "M": (0.0001, 0.08),
}
