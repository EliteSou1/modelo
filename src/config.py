"""Configuración compartida del pipeline CaféSmart AI (sin rutas absolutas)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORE_STATE_MAP: dict[str, str] = {
    "Astoria": "Quintana Roo",
    "Hell's Kitchen": "Ciudad de México",
    "Lower Manhattan": "Nuevo León",
}

CUTOFF_DATE = "2023-06-01"
BASE_START_DATE = "2023-01-01"
RANDOM_STATE = 42

CALENDAR_FEATURES = [
    "dia_semana",
    "es_fin_semana",
    "dia_mes",
    "mes",
    "semana_anio",
    "dia_indice",
    "sin_dia_semana",
    "cos_dia_semana",
    "es_quincena",
    "dias_para_fin_mes",
    "es_ultimos_dias_mes",
    "es_inicio_mes",
    "sin_dia_mes",
    "cos_dia_mes",
]

LAG_FEATURES = [
    "unidades_lag_1",
    "unidades_lag_2",
    "unidades_lag_3",
    "unidades_lag_7",
    "unidades_lag_14",
    "media_movil_7d",
    "media_movil_14d",
    "media_movil_28d",
    "std_movil_7d",
    "std_movil_14d",
    "tendencia_7d_28d",
    "ratio_7d_28d",
    "momentum_1d_7d",
    "media_mismo_dow_4s",
]

WEATHER_FEATURES = [
    "temp_min",
    "temp_media",
    "temp_max",
    "precipitacion",
    "amplitud_termica",
]

NUMERIC_FEATURES = CALENDAR_FEATURES + LAG_FEATURES + WEATHER_FEATURES

CATEGORICAL_FEATURES = [
    "store_location",
    "product_category",
    "estado_clima",
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "unidades_vendidas"
