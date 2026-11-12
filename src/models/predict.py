"""Módulo desacoplado de inferencia para CaféSmart AI (contrato UI Entrega 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import ALL_MODEL_FEATURES, BASE_START_DATE, PROJECT_ROOT, STORE_STATE_MAP

_MODELS_DIR = PROJECT_ROOT / "models"
MODELO = joblib.load(_MODELS_DIR / "modelo_candidato.joblib")
SCALER = joblib.load(_MODELS_DIR / "scaler.joblib")
_DEFAULTS: dict[str, Any] = json.loads((_MODELS_DIR / "feature_defaults.json").read_text(encoding="utf-8"))


def _build_feature_row(entrada: dict[str, Any]) -> tuple[pd.DataFrame, str, str, str, str]:
    """Valida entrada plana del usuario y construye el DataFrame exacto de entrada al pipeline."""
    if not isinstance(entrada, dict):
        raise TypeError("entrada debe ser un diccionario.")

    store = entrada.get("store_location")
    category = entrada.get("product_category")
    fecha_raw = entrada.get("fecha")

    if store not in STORE_STATE_MAP:
        raise ValueError(f"Sucursal no válida: {store!r}. Opciones: {sorted(STORE_STATE_MAP)}")
    if category not in _DEFAULTS["categorias_validas"]:
        raise ValueError(
            f"Categoría no válida: {category!r}. Opciones: {_DEFAULTS['categorias_validas']}"
        )
    if not fecha_raw:
        raise ValueError("El campo 'fecha' (YYYY-MM-DD) es obligatorio.")

    fecha_ts = pd.to_datetime(fecha_raw, format="%Y-%m-%d", errors="raise")
    base_ts = pd.Timestamp(BASE_START_DATE)
    estado = STORE_STATE_MAP[store]

    series_key = f"{store}||{category}"
    lag_ref = _DEFAULTS["series_defaults"].get(series_key, {})
    weather_key = f"{estado}||{fecha_ts.month}"
    weather_ref = _DEFAULTS["weather_defaults"].get(
        weather_key,
        {"temp_min": 18.0, "temp_media": 23.0, "temp_max": 29.0, "precipitacion": 25.0},
    )

    temp_min = float(entrada.get("temp_min", weather_ref["temp_min"]))
    temp_media = float(entrada.get("temp_media", weather_ref["temp_media"]))
    temp_max = float(entrada.get("temp_max", weather_ref["temp_max"]))
    precipitacion = float(entrada.get("precipitacion", weather_ref["precipitacion"]))
    if temp_max < temp_min or precipitacion < 0:
        raise ValueError("Variables meteorológicas inválidas (temp_max < temp_min o precipitación < 0).")

    media_7d = float(entrada.get("media_movil_7d", lag_ref.get("media_movil_7d", 0.0)))
    media_28d = float(entrada.get("media_movil_28d", lag_ref.get("media_movil_28d", 0.0)))
    lag_1 = float(entrada.get("unidades_lag_1", lag_ref.get("unidades_lag_1", 0.0)))
    lag_7 = float(entrada.get("unidades_lag_7", lag_ref.get("unidades_lag_7", 0.0)))
    dia_semana = int(fecha_ts.day_of_week)
    days_in_month = int(fecha_ts.days_in_month)
    dow_default = lag_ref.get("dow_defaults", {}).get(str(dia_semana), media_28d)

    row = {
        "dia_semana": dia_semana,
        "es_fin_semana": int(dia_semana >= 5),
        "dia_mes": int(fecha_ts.day),
        "mes": int(fecha_ts.month),
        "semana_anio": int(fecha_ts.isocalendar().week),
        "dia_indice": int((fecha_ts - base_ts).days),
        "sin_dia_semana": float(np.sin(2 * np.pi * dia_semana / 7.0)),
        "cos_dia_semana": float(np.cos(2 * np.pi * dia_semana / 7.0)),
        "es_quincena": int(fecha_ts.day in {1, 15, 16}),
        "dias_para_fin_mes": days_in_month - int(fecha_ts.day),
        "es_ultimos_dias_mes": int(fecha_ts.day >= 28),
        "es_inicio_mes": int(fecha_ts.day <= 3),
        "sin_dia_mes": float(np.sin(2 * np.pi * fecha_ts.day / days_in_month)),
        "cos_dia_mes": float(np.cos(2 * np.pi * fecha_ts.day / days_in_month)),
        "unidades_lag_1": lag_1,
        "unidades_lag_2": float(entrada.get("unidades_lag_2", lag_ref.get("unidades_lag_2", lag_1))),
        "unidades_lag_3": float(entrada.get("unidades_lag_3", lag_ref.get("unidades_lag_3", lag_1))),
        "unidades_lag_7": lag_7,
        "unidades_lag_14": float(entrada.get("unidades_lag_14", lag_ref.get("unidades_lag_14", 0.0))),
        "media_movil_7d": media_7d,
        "media_movil_14d": float(entrada.get("media_movil_14d", lag_ref.get("media_movil_14d", 0.0))),
        "media_movil_28d": media_28d,
        "std_movil_7d": float(entrada.get("std_movil_7d", lag_ref.get("std_movil_7d", 0.0))),
        "std_movil_14d": float(entrada.get("std_movil_14d", lag_ref.get("std_movil_14d", 0.0))),
        "tendencia_7d_28d": media_7d - media_28d,
        "ratio_7d_28d": (media_7d + 1.0) / (media_28d + 1.0),
        "momentum_1d_7d": lag_1 - lag_7,
        "media_mismo_dow_4s": float(entrada.get("media_mismo_dow_4s", dow_default)),
        "temp_min": temp_min,
        "temp_media": temp_media,
        "temp_max": temp_max,
        "precipitacion": precipitacion,
        "amplitud_termica": round(temp_max - temp_min, 2),
        "store_location": store,
        "product_category": category,
        "estado_clima": estado,
    }
    return pd.DataFrame([row], columns=ALL_MODEL_FEATURES), store, estado, category, fecha_ts.strftime("%Y-%m-%d")


def predecir(entrada: dict[str, Any]) -> dict[str, Any]:
    """
    Recibe las variables de entrada como diccionario plano (provenientes del formulario)
    y retorna la estimación estructurada de demanda diaria lista para el dashboard.
    """
    df_input, store, estado, category, fecha_str = _build_feature_row(entrada)
    X_scaled = SCALER.transform(df_input)
    prediccion = float(np.clip(MODELO.predict(X_scaled)[0], 0.0, None))

    mae_ref = float(_DEFAULTS["metricas_test_ganador"]["mae"])
    rmse_ref = float(_DEFAULTS["metricas_test_ganador"]["rmse"])
    margen_85 = max(1.0, 1.44 * rmse_ref * (0.35 if prediccion < 15 else 1.0))

    media_ref = float(_DEFAULTS["series_defaults"].get(f"{store}||{category}", {}).get("media_movil_28d", prediccion))
    if prediccion >= media_ref * 1.12 and prediccion >= 5:
        nivel = "Demanda Alta"
    elif prediccion <= media_ref * 0.88:
        nivel = "Demanda Baja"
    else:
        nivel = "Demanda Habitual"

    return {
        "sucursal": store,
        "estado_clima": estado,
        "categoria": category,
        "fecha": fecha_str,
        "unidades_estimadas": round(prediccion, 1),
        "unidades_recomendadas": int(np.ceil(prediccion)),
        "rango_estimado_85": [
            int(max(0, np.floor(prediccion - margen_85))),
            int(np.ceil(prediccion + margen_85)),
        ],
        "nivel_demanda": nivel,
        "modelo_usado": _DEFAULTS["modelo_ganador"],
        "error_esperado_mae": mae_ref,
        "status": "success",
    }


if __name__ == "__main__":
    muestra = {
        "store_location": "Astoria",
        "product_category": "Coffee",
        "fecha": "2023-07-03",
        "temp_media": 29.5,
        "precipitacion": 82.0,
    }
    print(json.dumps(predecir(muestra), indent=2, ensure_ascii=False))
