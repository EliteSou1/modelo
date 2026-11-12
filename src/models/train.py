"""Entrenamiento, comparación experimental y serialización de modelos CaféSmart AI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (
    ALL_MODEL_FEATURES,
    CALENDAR_FEATURES,
    CATEGORICAL_FEATURES,
    CUTOFF_DATE,
    LAG_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    RANDOM_STATE,
    STORE_STATE_MAP,
    TARGET_COL,
    WEATHER_FEATURES,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, categories: pd.Series | None = None) -> dict[str, float]:
    """Calcula métricas de regresión y pronóstico de demanda (WAPE, MAE, RMSE, R2, MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    total_actual = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100.0) if total_actual > 0 else 0.0

    # MAPE clásico solo es finito donde y_true > 0; calculamos además MAPE en las 4 categorías core (0 días en cero)
    nonzero_mask = y_true > 0
    mape_nonzero = (
        float(np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]) * 100.0)
        if np.any(nonzero_mask)
        else 0.0
    )
    mape_core = mape_nonzero
    if categories is not None:
        core_cats = {"Coffee", "Tea", "Bakery", "Drinking Chocolate"}
        core_mask = categories.isin(core_cats).to_numpy() & nonzero_mask
        if np.any(core_mask):
            mape_core = float(
                np.mean(np.abs(y_true[core_mask] - y_pred[core_mask]) / y_true[core_mask]) * 100.0
            )

    return {
        "wape": round(wape, 2),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape_activos": round(mape_nonzero, 2),
        "mape_core": round(mape_core, 2),
    }


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Construye el transformador de escalado numérico y codificación One-Hot."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )


def diagnose_fit(train_wape: float, test_wape: float, baseline_test_wape: float) -> str:
    """Diagnostica sobreajuste, subajuste o generalización saludable comparando Train vs Test."""
    gap = test_wape - train_wape
    if gap > 6.0 and train_wape < 10.0:
        return f"Overfitting moderado/alto (brecha Train-Test: {gap:.2f} pp)"
    if test_wape >= baseline_test_wape:
        return f"Subajuste / No supera Baseline ({baseline_test_wape:.2f}%)"
    if gap <= 3.5:
        return f"Generalización sólida (brecha Train-Test: {gap:.2f} pp)"
    return f"Buen ajuste con ligera varianza (brecha: {gap:.2f} pp)"


def get_feature_names(preprocessor: ColumnTransformer, num_cols: list[str], cat_cols: list[str]) -> list[str]:
    """Extrae los nombres transformados del ColumnTransformer."""
    ohe = preprocessor.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    return list(num_cols) + cat_names


def save_diagnostic_plots(
    results: list[dict],
    test_df: pd.DataFrame,
    winner_name: str,
    winner_pred: np.ndarray,
    baseline_pred: np.ndarray,
    feature_names: list[str],
    winner_model: object,
    figures_dir: Path,
) -> None:
    """Genera las 3 figuras diagnósticas para el reporte técnico de Entrega 3."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#CCCCCC"})

    # 1. Comparativa de modelos (WAPE % y RMSE en Test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    names = [r["experimento"] for r in results]
    wapes = [r["test"]["wape"] for r in results]
    rmses = [r["test"]["rmse"] for r in results]
    colors = ["#2E7D32" if r["experimento"] == winner_name else ("#8D6E63" if "Baseline" in r["experimento"] else "#546E7A") for r in results]

    y_pos = np.arange(len(names))
    axes[0].barh(y_pos, wapes, color=colors)
    axes[0].set_yticks(y_pos, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("WAPE en Test (%) — Menor es mejor")
    axes[0].set_title("Error Porcentual Ponderado (WAPE)")
    for i, v in enumerate(wapes):
        axes[0].text(v + 0.25, i, f"{v:.2f}%", va="center", fontsize=9)
    axes[0].set_xlim(0, max(wapes) * 1.18)

    axes[1].barh(y_pos, rmses, color=colors)
    axes[1].set_yticks(y_pos, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RMSE en Test (unidades/día) — Menor es mejor")
    axes[1].set_title("Raíz del Error Cuadrático Medio (RMSE)")
    for i, v in enumerate(rmses):
        axes[1].text(v + 0.25, i, f"{v:.2f}", va="center", fontsize=9)
    axes[1].set_xlim(0, max(rmses) * 1.18)

    fig.suptitle("CaféSmart AI — Comparativa Experimental en Test Set (Junio 2023)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "01_comparativa_modelos.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Serie temporal diaria agregada (Real vs Ganador vs Baseline) + Dispersión Real vs Predicho
    eval_df = test_df[["fecha", "store_location", "product_category", TARGET_COL]].copy()
    eval_df["pred_ganador"] = winner_pred
    eval_df["pred_baseline"] = baseline_pred
    daily_agg = eval_df.groupby("fecha", as_index=False)[[TARGET_COL, "pred_ganador", "pred_baseline"]].sum()
    daily_agg["fecha_dt"] = pd.to_datetime(daily_agg["fecha"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    axes[0].plot(daily_agg["fecha_dt"], daily_agg[TARGET_COL], label="Demanda Real", color="#22140D", linewidth=2.2, marker="o", markersize=3.5)
    axes[0].plot(daily_agg["fecha_dt"], daily_agg["pred_ganador"], label=f"Modelo Ganador ({winner_name})", color="#2E7D32", linewidth=2.0, linestyle="--")
    axes[0].plot(daily_agg["fecha_dt"], daily_agg["pred_baseline"], label="Baseline Naive (Lag-7)", color="#B0BEC5", linewidth=1.6, linestyle=":")
    axes[0].set_title("Demanda Diaria Total en Junio 2023 (3 Sucursales)")
    axes[0].set_ylabel("Unidades vendidas / día")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(frameon=True, fontsize=8.5)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(eval_df[TARGET_COL], eval_df["pred_ganador"], alpha=0.45, s=18, color="#4E342E", edgecolors="none")
    max_val = max(eval_df[TARGET_COL].max(), eval_df["pred_ganador"].max()) * 1.05
    axes[1].plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Predicción ideal (y = x)")
    r2_val = r2_score(eval_df[TARGET_COL], eval_df["pred_ganador"])
    axes[1].set_title(f"Dispersión Real vs Predicho (Test Set, R² = {r2_val:.4f})")
    axes[1].set_xlabel("Unidades Reales")
    axes[1].set_ylabel("Unidades Predichas")
    axes[1].legend(frameon=True, fontsize=8.5)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(figures_dir / "02_curva_real_vs_predicho_junio.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Matriz de error MAE por Sucursal x Categoría + Top 10 Importancia de Variables
    eval_df["abs_err"] = np.abs(eval_df[TARGET_COL] - eval_df["pred_ganador"])
    err_matrix = eval_df.pivot_table(index="product_category", columns="store_location", values="abs_err", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
    im = axes[0].imshow(err_matrix.values, cmap="YlOrBr", aspect="auto")
    axes[0].set_xticks(range(len(err_matrix.columns)), err_matrix.columns, rotation=15)
    axes[0].set_yticks(range(len(err_matrix.index)), err_matrix.index)
    axes[0].set_title("Matriz de Error MAE (uds/día) por Sucursal × Categoría")
    for r in range(err_matrix.shape[0]):
        for c in range(err_matrix.shape[1]):
            val = err_matrix.values[r, c]
            txt_color = "white" if val > err_matrix.values.max() * 0.6 else "#22140D"
            axes[0].text(c, r, f"{val:.1f}", ha="center", va="center", color=txt_color, fontsize=8.5, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    if hasattr(winner_model, "feature_importances_"):
        importances = np.asarray(winner_model.feature_importances_, dtype=float)
    elif hasattr(winner_model, "coef_"):
        importances = np.abs(np.asarray(winner_model.coef_, dtype=float))
    else:
        importances = np.zeros(len(feature_names))

    top_idx = np.argsort(importances)[-10:]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importances[top_idx]
    if top_vals.sum() > 0:
        top_vals = top_vals / top_vals.sum() * 100.0
    axes[1].barh(range(len(top_names)), top_vals, color="#6D4C41")
    axes[1].set_yticks(range(len(top_names)), top_names, fontsize=8.5)
    axes[1].set_xlabel("Importancia relativa (%)")
    axes[1].set_title(f"Top 10 Variables Influyentes ({winner_name})")
    for i, v in enumerate(top_vals):
        axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[1].set_xlim(0, max(top_vals) * 1.18 if len(top_vals) else 1)

    fig.tight_layout()
    fig.savefig(figures_dir / "03_analisis_error_y_variables.png", bbox_inches="tight")
    plt.close(fig)


def build_feature_defaults(df: pd.DataFrame, winner_metrics: dict[str, float], winner_name: str) -> dict:
    """Calcula referencias históricas por sucursal/categoría y promedios climáticos para inferencia."""
    df_sorted = df.sort_values(["store_location", "product_category", "fecha"]).reset_index(drop=True)
    recent_cut = pd.to_datetime(df_sorted["fecha"]).max() - pd.Timedelta(days=27)
    recent = df_sorted[pd.to_datetime(df_sorted["fecha"]) >= recent_cut]

    series_defaults: dict[str, dict] = {}
    full_groups = dict(tuple(df_sorted.groupby(["store_location", "product_category"])))
    for (store, cat), grp in recent.groupby(["store_location", "product_category"]):
        key = f"{store}||{cat}"
        full_grp = full_groups[(store, cat)]
        dow_defaults = {
            str(int(dow)): round(float(sub["unidades_vendidas"].tail(4).mean()), 2)
            for dow, sub in full_grp.groupby("dia_semana")
        }
        series_defaults[key] = {
            "unidades_lag_1": round(float(grp["unidades_vendidas"].iloc[-1]), 2),
            "unidades_lag_2": round(float(grp["unidades_vendidas"].iloc[-2]) if len(grp) >= 2 else float(grp["unidades_vendidas"].mean()), 2),
            "unidades_lag_3": round(float(grp["unidades_vendidas"].iloc[-3]) if len(grp) >= 3 else float(grp["unidades_vendidas"].mean()), 2),
            "unidades_lag_7": round(float(grp["unidades_vendidas"].iloc[-7]) if len(grp) >= 7 else float(grp["unidades_vendidas"].mean()), 2),
            "unidades_lag_14": round(float(grp["unidades_vendidas"].iloc[-14]) if len(grp) >= 14 else float(grp["unidades_vendidas"].mean()), 2),
            "media_movil_7d": round(float(grp["unidades_vendidas"].tail(7).mean()), 2),
            "media_movil_14d": round(float(grp["unidades_vendidas"].tail(14).mean()), 2),
            "media_movil_28d": round(float(grp["unidades_vendidas"].mean()), 2),
            "std_movil_7d": round(float(grp["unidades_vendidas"].tail(7).std(ddof=1)) if len(grp) >= 2 else 0.0, 2),
            "std_movil_14d": round(float(grp["unidades_vendidas"].tail(14).std(ddof=1)) if len(grp) >= 2 else 0.0, 2),
            "dow_defaults": dow_defaults,
        }

    weather_defaults: dict[str, dict[str, float]] = {}
    for (estado, mes), grp in df.groupby(["estado_clima", "mes"]):
        weather_defaults[f"{estado}||{int(mes)}"] = {
            "temp_min": round(float(grp["temp_min"].mean()), 2),
            "temp_media": round(float(grp["temp_media"].mean()), 2),
            "temp_max": round(float(grp["temp_max"].mean()), 2),
            "precipitacion": round(float(grp["precipitacion"].mean()), 2),
        }

    return {
        "modelo_ganador": winner_name,
        "metricas_test_ganador": winner_metrics,
        "sucursales_estados": STORE_STATE_MAP,
        "categorias_validas": sorted(df["product_category"].unique().tolist()),
        "series_defaults": series_defaults,
        "weather_defaults": weather_defaults,
    }


def run_experiments(data_path: Path, output_dir: Path, figures_dir: Path) -> None:
    """Ejecuta entrenamiento comparativo, ablación climática, diagnóstico y serialización."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset limpio no encontrado: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[INFO] Cargando dataset: {data_path} ({len(df):,} registros)...")

    train_mask = df["fecha"] < CUTOFF_DATE
    test_mask = df["fecha"] >= CUTOFF_DATE
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    print(
        f"[INFO] Partición cronológica (corte {CUTOFF_DATE}) | "
        f"Train: {len(train_df):,} muestras ({train_df['fecha'].min()} a {train_df['fecha'].max()}) | "
        f"Test: {len(test_df):,} muestras ({test_df['fecha'].min()} a {test_df['fecha'].max()})"
    )

    y_train = train_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    # 1. Baseline Naive Estacional (misma sucursal/categoría 7 días atrás)
    t0 = time.perf_counter()
    base_train_pred = train_df["unidades_lag_7"].to_numpy()
    base_test_pred = test_df["unidades_lag_7"].to_numpy()
    base_time = time.perf_counter() - t0
    base_train_m = compute_metrics(y_train, base_train_pred, train_df["product_category"])
    base_test_m = compute_metrics(y_test, base_test_pred, test_df["product_category"])

    results: list[dict] = [
        {
            "experimento": "Exp 0: Baseline Naive (Lag-7)",
            "modelo": "Naive Estacional 7d",
            "hiperparametros": "y_hat(t) = y(t - 7)",
            "train": base_train_m,
            "test": base_test_m,
            "tiempo_train_s": round(base_time, 4),
            "tiempo_inf_ms": round(base_time * 1000, 3),
            "diagnostico": "Referencia base (sin entrenamiento)",
            "usa_clima": False,
        }
    ]

    # 2. Preprocesador completo (con clima) ajustado SOLO en Train
    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X_train_proc = preprocessor.fit_transform(train_df[ALL_MODEL_FEATURES])
    X_test_proc = preprocessor.transform(test_df[ALL_MODEL_FEATURES])
    feature_names = get_feature_names(preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    candidates = [
        (
            "Exp 1: Ridge L2",
            "Ridge Regression",
            "alpha=10.0",
            Ridge(alpha=10.0),
        ),
        (
            "Exp 2: RF Base (Sin poda)",
            "Random Forest Regressor",
            "n_estimators=200, max_depth=None, min_samples_leaf=1",
            RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1),
        ),
        (
            "Exp 3: RF Regularizado",
            "Random Forest Regressor",
            "n_estimators=300, max_depth=10, min_samples_leaf=4, max_features=0.8",
            RandomForestRegressor(
                n_estimators=300, max_depth=10, min_samples_leaf=4, max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
        (
            "Exp 4: XGBoost Tweedie Reg.",
            "XGBRegressor (Tweedie)",
            "objective=reg:tweedie(1.4), n_estimators=500, max_depth=5, lr=0.03, reg_lambda=2.0",
            XGBRegressor(
                objective="reg:tweedie",
                tweedie_variance_power=1.4,
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "Exp 5: LightGBM Regularizado (L2)",
            "LGBMRegressor (L2)",
            "n_estimators=300, num_leaves=20, max_depth=5, lr=0.05, min_child=15, reg_lambda=2.0",
            LGBMRegressor(
                n_estimators=300,
                num_leaves=20,
                max_depth=5,
                learning_rate=0.05,
                min_child_samples=15,
                reg_alpha=0.5,
                reg_lambda=2.0,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1,
            ),
        ),
        (
            "Exp 6: LightGBM Tweedie Reg.",
            "LGBMRegressor (Tweedie)",
            "objective=tweedie(1.4), n_estimators=400, num_leaves=24, max_depth=6, lr=0.04, min_child=15",
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.4,
                n_estimators=400,
                num_leaves=24,
                max_depth=6,
                learning_rate=0.04,
                min_child_samples=15,
                reg_alpha=0.5,
                reg_lambda=2.0,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1,
            ),
        ),
    ]

    trained_models: dict[str, tuple[object, np.ndarray]] = {}
    for exp_name, model_label, hp_desc, model in candidates:
        print(f"[INFO] Entrenando {exp_name} ({model_label})...")
        t_start = time.perf_counter()
        model.fit(X_train_proc, y_train)
        train_s = time.perf_counter() - t_start

        t_inf_start = time.perf_counter()
        test_pred = np.clip(model.predict(X_test_proc), 0.0, None)
        inf_ms = (time.perf_counter() - t_inf_start) * 1000.0
        train_pred = np.clip(model.predict(X_train_proc), 0.0, None)

        train_m = compute_metrics(y_train, train_pred, train_df["product_category"])
        test_m = compute_metrics(y_test, test_pred, test_df["product_category"])
        diag = diagnose_fit(train_m["wape"], test_m["wape"], base_test_m["wape"])

        results.append(
            {
                "experimento": exp_name,
                "modelo": model_label,
                "hiperparametros": hp_desc,
                "train": train_m,
                "test": test_m,
                "tiempo_train_s": round(train_s, 3),
                "tiempo_inf_ms": round(inf_ms, 2),
                "diagnostico": diag,
                "usa_clima": True,
            }
        )
        trained_models[exp_name] = (model, test_pred)

    # Seleccionar mejor modelo ML por menor WAPE en Test (desempate por RMSE)
    ml_results = [r for r in results if r["experimento"] != "Exp 0: Baseline Naive (Lag-7)"]
    best_entry = min(ml_results, key=lambda r: (r["test"]["wape"], r["test"]["rmse"]))
    winner_name = best_entry["experimento"]
    winner_model, winner_pred = trained_models[winner_name]

    # 3. Experimento de ablación: misma arquitectura ganadora SIN variables meteorológicas
    no_weather_num = CALENDAR_FEATURES + LAG_FEATURES
    no_weather_cat = ["store_location", "product_category"]
    pre_noweather = build_preprocessor(no_weather_num, no_weather_cat)
    X_train_nw = pre_noweather.fit_transform(train_df[no_weather_num + no_weather_cat])
    X_test_nw = pre_noweather.transform(test_df[no_weather_num + no_weather_cat])

    from sklearn.base import clone
    ablation_model = clone(winner_model)
    t_ab = time.perf_counter()
    ablation_model.fit(X_train_nw, y_train)
    ab_train_s = time.perf_counter() - t_ab
    t_ab_inf = time.perf_counter()
    ab_test_pred = np.clip(ablation_model.predict(X_test_nw), 0.0, None)
    ab_inf_ms = (time.perf_counter() - t_ab_inf) * 1000.0
    ab_train_pred = np.clip(ablation_model.predict(X_train_nw), 0.0, None)

    ab_train_m = compute_metrics(y_train, ab_train_pred, train_df["product_category"])
    ab_test_m = compute_metrics(y_test, ab_test_pred, test_df["product_category"])
    results.append(
        {
            "experimento": f"Exp 7: Ablación sin Clima ({best_entry['modelo']})",
            "modelo": f"{best_entry['modelo']} (Sin Clima)",
            "hiperparametros": f"{best_entry['hiperparametros']} | sin temp/precipitación",
            "train": ab_train_m,
            "test": ab_test_m,
            "tiempo_train_s": round(ab_train_s, 3),
            "tiempo_inf_ms": round(ab_inf_ms, 2),
            "diagnostico": diagnose_fit(ab_train_m["wape"], ab_test_m["wape"], base_test_m["wape"]),
            "usa_clima": False,
        }
    )

    # Imprimir tabla resumen en consola
    print("\n[RESULTADOS EN TEST SET - JUNIO 2023 (810 registros)]")
    print(f"{'Experimento':<38} | {'WAPE(%)':>7} | {'MAE':>6} | {'RMSE':>6} | {'R2':>7} | {'MAPE_core%':>10} | {'Train_WAPE%':>11} | {'Tiempo(s)':>9}")
    print("-" * 115)
    for r in results:
        t_m, tr_m = r["test"], r["train"]
        print(
            f"{r['experimento']:<38} | {t_m['wape']:7.2f} | {t_m['mae']:6.2f} | {t_m['rmse']:6.2f} | "
            f"{t_m['r2']:7.4f} | {t_m['mape_core']:10.2f} | {tr_m['wape']:11.2f} | {r['tiempo_train_s']:9.3f}"
        )

    improvement_vs_base = (base_test_m["wape"] - best_entry["test"]["wape"]) / base_test_m["wape"] * 100.0
    print(
        f"\n[GANADOR] {winner_name} ({best_entry['modelo']}) -> "
        f"Test WAPE={best_entry['test']['wape']:.2f}% | MAE={best_entry['test']['mae']:.2f} uds | "
        f"RMSE={best_entry['test']['rmse']:.2f} uds | R2={best_entry['test']['r2']:.4f} | "
        f"Mejora relativa vs Baseline: {improvement_vs_base:.2f}%"
    )

    # Serializar artefactos
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "modelo_candidato.joblib"
    scaler_file = output_dir / "scaler.joblib"
    defaults_file = output_dir / "feature_defaults.json"
    results_file = output_dir / "experiment_results.json"

    joblib.dump(winner_model, model_file)
    joblib.dump(preprocessor, scaler_file)

    defaults_payload = build_feature_defaults(df, best_entry["test"], best_entry["modelo"])
    defaults_file.write_text(json.dumps(defaults_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_payload = {
        "cutoff_date": CUTOFF_DATE,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "winner_experiment": winner_name,
        "improvement_vs_baseline_pct": round(improvement_vs_base, 2),
        "weather_delta_wape_pp": round(ab_test_m["wape"] - best_entry["test"]["wape"], 2),
        "experiments": results,
    }
    results_file.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    save_diagnostic_plots(
        results=results,
        test_df=test_df,
        winner_name=winner_name,
        winner_pred=winner_pred,
        baseline_pred=base_test_pred,
        feature_names=feature_names,
        winner_model=winner_model,
        figures_dir=figures_dir,
    )

    print(f"[SUCCESS] Modelo serializado con éxito en: {model_file}")
    print(f"[SUCCESS] Preprocesador/Escalador guardado en: {scaler_file}")
    print(f"[SUCCESS] Gráficas diagnósticas guardadas en: {figures_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento y comparación CaféSmart AI")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "dataset_limpio.csv",
        help="Ruta al CSV procesado",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directorio donde guardar modelos y resultados",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures",
        help="Directorio para figuras diagnósticas",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args.data, args.output, args.figures)
