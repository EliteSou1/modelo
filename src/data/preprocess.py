"""Pipeline reproducible de limpieza, agregación e integración climática para CaféSmart AI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import PROJECT_ROOT, STORE_STATE_MAP


def load_and_clean_sales(sales_path: Path) -> tuple[pd.DataFrame, int]:
    """Carga Coffee Shop Sales.csv, valida integridad y agrega ventas diarias."""
    if not sales_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de ventas: {sales_path}")

    df = pd.read_excel(sales_path) if sales_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(sales_path)
    required = {
        "transaction_id",
        "transaction_date",
        "transaction_qty",
        "store_location",
        "product_category",
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas obligatorias en ventas: {sorted(missing_cols)}")

    initial_rows = len(df)
    df = df.drop_duplicates(subset=["transaction_id"]).dropna(
        subset=["transaction_date", "transaction_qty", "store_location", "product_category"]
    )
    df = df[df["transaction_qty"] > 0].copy()
    if pd.api.types.is_datetime64_any_dtype(df["transaction_date"]):
        df["fecha"] = pd.to_datetime(df["transaction_date"]).dt.normalize()
    else:
        df["fecha"] = pd.to_datetime(df["transaction_date"], format="%m/%d/%y")

    unknown_stores = set(df["store_location"].unique()) - set(STORE_STATE_MAP)
    if unknown_stores:
        raise ValueError(f"Sucursales sin mapeo climático: {sorted(unknown_stores)}")

    raw_total_units = int(df["transaction_qty"].sum())
    print(
        f"[INFO] Ventas cargadas: {initial_rows:,} transacciones | "
        f"Válidas: {len(df):,} ({len(df) / initial_rows * 100:.2f}%) | "
        f"Unidades totales: {raw_total_units:,}"
    )

    daily = (
        df.groupby(["fecha", "store_location", "product_category"], as_index=False)["transaction_qty"]
        .sum()
        .rename(columns={"transaction_qty": "unidades_vendidas"})
    )

    # Producto cartesiano completo para incluir explícitamente días con 0 ventas (sin sesgo de supervivencia)
    all_dates = pd.date_range(daily["fecha"].min(), daily["fecha"].max(), freq="D")
    stores = sorted(STORE_STATE_MAP.keys())
    categories = sorted(df["product_category"].unique())

    grid = pd.MultiIndex.from_product(
        [all_dates, stores, categories],
        names=["fecha", "store_location", "product_category"],
    ).to_frame(index=False)

    daily_full = grid.merge(daily, on=["fecha", "store_location", "product_category"], how="left")
    daily_full["unidades_vendidas"] = daily_full["unidades_vendidas"].fillna(0).astype(int)

    grid_total_units = int(daily_full["unidades_vendidas"].sum())
    if grid_total_units != raw_total_units:
        raise AssertionError(
            f"Fallo de conservación de unidades: origen={raw_total_units} vs grid={grid_total_units}"
        )

    daily_full["estado_clima"] = daily_full["store_location"].map(STORE_STATE_MAP)
    print(
        f"[INFO] Matriz diaria completa: {len(daily_full):,} filas "
        f"({len(all_dates)} días × {len(stores)} sucursales × {len(categories)} categorías) | "
        f"Diferencia de unidades: 0"
    )
    return daily_full, raw_total_units


def load_weather(weather_path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Carga el dataset meteorológico estatal de México y filtra las entidades mapeadas."""
    if not weather_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo meteorológico: {weather_path}")

    wdf = pd.read_excel(weather_path)
    required = {"PERIODO", "ENTIDAD", "MINIMA", "MEDIA", "MAXIMA", "PRECIPITACION"}
    missing_cols = required - set(wdf.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas en clima: {sorted(missing_cols)}")

    if pd.api.types.is_numeric_dtype(wdf["PERIODO"]):
        wdf["periodo_fecha"] = pd.to_datetime(wdf["PERIODO"], unit="D", origin="1899-12-30")
    else:
        wdf["periodo_fecha"] = pd.to_datetime(wdf["PERIODO"])

    wdf["anio"] = wdf["periodo_fecha"].dt.year
    wdf["mes"] = wdf["periodo_fecha"].dt.month

    target_states = set(STORE_STATE_MAP.values())
    wdf = wdf[
        (wdf["ENTIDAD"].isin(target_states))
        & (wdf["periodo_fecha"] >= start_date.to_period("M").to_timestamp())
        & (wdf["periodo_fecha"] <= end_date.to_period("M").to_timestamp())
    ].copy()

    wdf = wdf.rename(
        columns={
            "ENTIDAD": "estado_clima",
            "MINIMA": "temp_min",
            "MEDIA": "temp_media",
            "MAXIMA": "temp_max",
            "PRECIPITACION": "precipitacion",
        }
    )[["estado_clima", "anio", "mes", "temp_min", "temp_media", "temp_max", "precipitacion"]]

    for col in ["temp_min", "temp_media", "temp_max", "precipitacion"]:
        wdf[col] = pd.to_numeric(wdf[col], errors="raise")

    wdf["amplitud_termica"] = (wdf["temp_max"] - wdf["temp_min"]).round(2)
    return wdf.drop_duplicates(subset=["estado_clima", "anio", "mes"])


def add_temporal_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye atributos de calendario y rezagos causales (shift >= 1) sin fuga temporal."""
    df = df.sort_values(["store_location", "product_category", "fecha"]).reset_index(drop=True)
    min_date = df["fecha"].min()

    df["dia_semana"] = df["fecha"].dt.day_of_week
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["dia_mes"] = df["fecha"].dt.day
    days_in_month = df["fecha"].dt.days_in_month
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["semana_anio"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["dia_indice"] = (df["fecha"] - min_date).dt.days
    df["sin_dia_semana"] = np.sin(2 * np.pi * df["dia_semana"] / 7.0)
    df["cos_dia_semana"] = np.cos(2 * np.pi * df["dia_semana"] / 7.0)
    df["es_quincena"] = df["dia_mes"].isin([1, 15, 16]).astype(int)
    df["dias_para_fin_mes"] = days_in_month - df["dia_mes"]
    df["es_ultimos_dias_mes"] = (df["dia_mes"] >= 28).astype(int)
    df["es_inicio_mes"] = (df["dia_mes"] <= 3).astype(int)
    df["sin_dia_mes"] = np.sin(2 * np.pi * df["dia_mes"] / days_in_month)
    df["cos_dia_mes"] = np.cos(2 * np.pi * df["dia_mes"] / days_in_month)

    grp = df.groupby(["store_location", "product_category"])["unidades_vendidas"]
    df["unidades_lag_1"] = grp.shift(1)
    df["unidades_lag_2"] = grp.shift(2)
    df["unidades_lag_3"] = grp.shift(3)
    df["unidades_lag_7"] = grp.shift(7)
    df["unidades_lag_14"] = grp.shift(14)

    shifted = grp.shift(1)
    series_keys = [df["store_location"], df["product_category"]]
    df["media_movil_7d"] = shifted.groupby(series_keys).transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["media_movil_14d"] = shifted.groupby(series_keys).transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["media_movil_28d"] = shifted.groupby(series_keys).transform(lambda s: s.rolling(28, min_periods=1).mean())
    df["std_movil_7d"] = (
        shifted.groupby(series_keys).transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0.0)
    )
    df["std_movil_14d"] = (
        shifted.groupby(series_keys).transform(lambda s: s.rolling(14, min_periods=1).std()).fillna(0.0)
    )
    df["tendencia_7d_28d"] = df["media_movil_7d"] - df["media_movil_28d"]
    df["ratio_7d_28d"] = (df["media_movil_7d"] + 1.0) / (df["media_movil_28d"] + 1.0)
    df["momentum_1d_7d"] = df["unidades_lag_1"] - df["unidades_lag_7"]
    df["media_mismo_dow_4s"] = (
        df.groupby(["store_location", "product_category", "dia_semana"])["unidades_vendidas"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    )
    return df


def build_processed_dataset(sales_path: Path, weather_path: Path, output_path: Path) -> pd.DataFrame:
    """Ejecuta el pipeline completo de preparación y guarda el CSV procesado."""
    daily_full, _ = load_and_clean_sales(sales_path)
    weather_df = load_weather(weather_path, daily_full["fecha"].min(), daily_full["fecha"].max())

    daily_full["anio"] = daily_full["fecha"].dt.year
    daily_full["mes"] = daily_full["fecha"].dt.month

    merged = daily_full.merge(weather_df, on=["estado_clima", "anio", "mes"], how="left")
    weather_cov = (1.0 - merged["temp_media"].isna().mean()) * 100.0
    print(f"[INFO] Cobertura del cruce meteorológico (3 sucursales -> 3 estados MX): {weather_cov:.1f}%")
    if weather_cov < 100.0:
        raise ValueError(f"Cobertura climática incompleta: {weather_cov:.2f}%")

    featured = add_temporal_and_lag_features(merged)
    # Descartar primeros 14 días de calentamiento para que todos los rezagos estén completos sin NaNs
    model_df = featured.dropna(subset=["unidades_lag_14", "media_mismo_dow_4s"]).copy()
    model_df["fecha"] = model_df["fecha"].dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(output_path, index=False)
    print(
        f"[SUCCESS] Dataset limpio guardado en: {output_path} | "
        f"Filas modelables: {len(model_df):,} | Columnas: {model_df.shape[1]} | Nulos: {int(model_df.isna().sum().sum())}"
    )
    return model_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesamiento reproducible CaféSmart AI")
    parser.add_argument(
        "--sales",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "Coffee Shop Sales.csv",
        help="Ruta relativa al CSV crudo de ventas",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "clima_mexico_estados.xlsx",
        help="Ruta relativa al XLSX meteorológico por estado",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "dataset_limpio.csv",
        help="Ruta de salida para el CSV limpio",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_processed_dataset(args.sales, args.weather, args.output)
