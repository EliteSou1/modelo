"""Pruebas automatizadas del pipeline CaféSmart AI (datos, clima, cero fuga, modelo, inferencia y entregable)."""

from __future__ import annotations

import json
import time
import zipfile

import docx
import joblib
import pandas as pd
import pytest

from src.config import CUTOFF_DATE, PROJECT_ROOT, STORE_STATE_MAP
from src.data.preprocess import load_and_clean_sales, load_weather
from src.models.predict import predecir


def test_processed_dataset_integrity() -> None:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_limpio.csv")
    assert df.shape == (4509, 39)
    assert df.isna().sum().sum() == 0
    assert (df["unidades_vendidas"] >= 0).all()
    assert set(df["store_location"].unique()) == set(STORE_STATE_MAP.keys())
    assert df["product_category"].nunique() == 9

    # Validar cruce meteorológico 100% y consistencia física de temperaturas/lluvia
    mapped_states = df["store_location"].map(STORE_STATE_MAP)
    assert (df["estado_clima"] == mapped_states).all()
    assert (df["temp_min"] <= df["temp_media"]).all()
    assert (df["temp_media"] <= df["temp_max"]).all()
    assert (df["precipitacion"] >= 0).all()

    # RNF-04: sin identificadores personales ni transaccionales individuales
    forbidden_cols = {"transaction_id", "customer_id", "nombre", "email", "telefono"}
    assert forbidden_cols.isdisjoint(set(df.columns))


def test_chronological_split_and_zero_leakage() -> None:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_limpio.csv")
    train_df = df[df["fecha"] < CUTOFF_DATE]
    test_df = df[df["fecha"] >= CUTOFF_DATE]
    assert len(train_df) == 3699
    assert len(test_df) == 810
    assert train_df["fecha"].max() < test_df["fecha"].min()

    # Verificar causalidad estricta de rezagos (t - 1, t - 2 y t - 7) en cada serie
    df_sorted = df.sort_values(["store_location", "product_category", "fecha"]).reset_index(drop=True)
    grp = df_sorted.groupby(["store_location", "product_category"])["unidades_vendidas"]
    s1, s2, s7 = grp.shift(1), grp.shift(2), grp.shift(7)
    m1, m2, m7 = s1.notna(), s2.notna(), s7.notna()
    assert (df_sorted.loc[m1, "unidades_lag_1"] == s1.loc[m1]).all()
    assert (df_sorted.loc[m2, "unidades_lag_2"] == s2.loc[m2]).all()
    assert (df_sorted.loc[m7, "unidades_lag_7"] == s7.loc[m7]).all()


def test_raw_unit_conservation_and_weather_coverage() -> None:
    sales_path = PROJECT_ROOT / "data" / "raw" / "Coffee Shop Sales.csv"
    weather_path = PROJECT_ROOT / "data" / "raw" / "clima_mexico_estados.xlsx"

    daily_full, raw_total_units = load_and_clean_sales(sales_path)
    assert raw_total_units == 214470
    assert len(daily_full) == 181 * 3 * 9
    assert int(daily_full["unidades_vendidas"].sum()) == raw_total_units

    weather_df = load_weather(weather_path, daily_full["fecha"].min(), daily_full["fecha"].max())
    assert len(weather_df) == 3 * 6
    assert set(weather_df["estado_clima"].unique()) == set(STORE_STATE_MAP.values())


def test_training_metrics_and_serialized_artifacts() -> None:
    models_dir = PROJECT_ROOT / "models"
    results = json.loads((models_dir / "experiment_results.json").read_text(encoding="utf-8"))
    experiments = results["experiments"]
    assert len(experiments) >= 3
    assert results["improvement_vs_baseline_pct"] >= 10.0

    winner = next(e for e in experiments if e["experimento"] == results["winner_experiment"])
    assert winner["test"]["mape_core"] <= 20.0
    assert winner["test"]["wape"] < experiments[0]["test"]["wape"]

    for fname in ["modelo_candidato.joblib", "scaler.joblib", "feature_defaults.json"]:
        assert (models_dir / fname).exists()
    assert joblib.load(models_dir / "modelo_candidato.joblib") is not None
    assert joblib.load(models_dir / "scaler.joblib") is not None

    figures_dir = PROJECT_ROOT / "reports" / "figures"
    for fig in [
        "01_comparativa_modelos.png",
        "02_curva_real_vs_predicho_junio.png",
        "03_analisis_error_y_variables.png",
    ]:
        fig_path = figures_dir / fig
        assert fig_path.exists() and fig_path.stat().st_size > 10_000


def test_predecir_all_stores_and_categories_within_sla() -> None:
    defaults = json.loads((PROJECT_ROOT / "models" / "feature_defaults.json").read_text(encoding="utf-8"))
    for store, expected_state in STORE_STATE_MAP.items():
        for category in defaults["categorias_validas"]:
            t0 = time.perf_counter()
            res = predecir(
                {
                    "store_location": store,
                    "product_category": category,
                    "fecha": "2023-07-03",
                }
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert res["status"] == "success"
            assert res["estado_clima"] == expected_state
            assert res["unidades_estimadas"] >= 0
            assert res["rango_estimado_85"][0] <= res["unidades_recomendadas"] <= res["rango_estimado_85"][1]
            assert res["nivel_demanda"] in {"Demanda Alta", "Demanda Baja", "Demanda Habitual"}
            assert elapsed_ms < 5000.0

    # RNF-02: determinismo en llamadas repetidas
    sample = {
        "store_location": "Astoria",
        "product_category": "Coffee",
        "fecha": "2023-07-03",
        "temp_media": 29.5,
        "precipitacion": 82.0,
    }
    assert predecir(sample) == predecir(sample)


@pytest.mark.parametrize(
    ("payload", "exc_type"),
    [
        ({"store_location": "Sucursal Inexistente", "product_category": "Coffee", "fecha": "2023-07-03"}, ValueError),
        ({"store_location": "Astoria", "product_category": "Categoria Invalida", "fecha": "2023-07-03"}, ValueError),
        ({"store_location": "Astoria", "product_category": "Coffee", "fecha": "03/07/2023"}, ValueError),
        (
            {
                "store_location": "Astoria",
                "product_category": "Coffee",
                "fecha": "2023-07-03",
                "temp_min": 32.0,
                "temp_max": 20.0,
            },
            ValueError,
        ),
        (
            {
                "store_location": "Astoria",
                "product_category": "Coffee",
                "fecha": "2023-07-03",
                "precipitacion": -5.0,
            },
            ValueError,
        ),
        ("no-es-dict", TypeError),
    ],
)
def test_predecir_rejects_invalid_inputs(payload: object, exc_type: type[Exception]) -> None:
    with pytest.raises(exc_type):
        predecir(payload)  # type: ignore[arg-type]


def test_entrega3_docx_integrity() -> None:
    doc_path = PROJECT_ROOT / "docs" / "ENTREGA_3_CAFESMART_AI.docx"
    assert doc_path.exists()
    doc = docx.Document(doc_path)
    assert len(doc.tables) == 12

    forbidden = ["____", "clientes_churn", "svm_churn", "PENDIENTE", "\ufeff"]
    full_text = "\n".join(p.text for p in doc.paragraphs)
    for tbl in doc.tables:
        for row in tbl.rows:
            full_text += "\n" + "\n".join(cell.text for cell in row.cells)
    for token in forbidden:
        assert token not in full_text

    with zipfile.ZipFile(doc_path) as zf:
        media = [n for n in zf.namelist() if n.startswith("word/media/")]
    assert len(media) == 3
