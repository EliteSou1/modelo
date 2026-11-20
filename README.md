
# CaféSmart AI — Sprint 3: Motor de Pronóstico de Demanda

Motor predictivo de demanda diaria por sucursal y categoría de producto, desarrollado para la materia **Fundamentos de Inteligencia Artificial** (Grupo **LITIID 72**, Universidad Tecnológica de Cancún).

## Equipo
- **Carlos Daniel Cocom García** — Líder de Datos y Preprocesamiento
- **Ángel Fernando Ocampo García 🥺🥺** — Líder de Modelado y Experimentación
- **Saúl Alejandro Rendón Cocom** — Líder de Pipeline e Integración
- **Héctor Zavala López** — PM / Validación y Documentación

## Mapeo Geográfico de Clima (Dataset Estatal de México)
Siguiendo la indicación del curso, cada sucursal del dataset `Coffee Shop Sales.csv` se asocia a una entidad federativa distinta del dataset meteorológico oficial (`clima_mexico_estados.xlsx`):
- `Astoria` → **Quintana Roo**
- `Hell's Kitchen` → **Ciudad de México**
- `Lower Manhattan` → **Nuevo León**

## Reproducibilidad
```bash
# 1. Preprocesar datos crudos (ventas + clima estatal)
python src/data/preprocess.py --sales "data/raw/Coffee Shop Sales.csv" --weather data/raw/clima_mexico_estados.xlsx --output data/processed/dataset_limpio.csv

# 2. Entrenar, comparar modelos, generar gráficas y serializar el modelo ganador
python src/models/train.py --data data/processed/dataset_limpio.csv --output models/

# 3. Probar función de inferencia desacoplada y correr suite de pruebas
python src/models/predict.py
pytest tests/test_pipeline.py
```
