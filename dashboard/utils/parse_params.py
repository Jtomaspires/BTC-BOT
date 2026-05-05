from __future__ import annotations


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]


def scalar_float_from_yaml(value: object, default: float) -> float:
    """``config.resolved.yaml`` pode guardar SL/TP como escalar ou lista (grelha); usa o 1.º elemento."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return float(value[0])
    return float(value)