# Training Models - Predicción de Precios de NVIDIA

Proyecto de modelos de machine learning para la predicción de precios de acciones de NVIDIA utilizando análisis de sentimiento y datos históricos.

## 📋 Descripción

Este proyecto implementa y compara tres modelos de machine learning para predecir movimientos en el precio de las acciones de NVIDIA:

- **LSTM (Long Short-Term Memory)**: Red neuronal recurrente para series temporales
- **Random Forest**: Modelo de ensamble basado en árboles de decisión
- **XGBoost**: Gradient boosting optimizado con Optuna

## 🗂️ Estructura del Proyecto

```
training_models/
├── data/                           # Datasets
│   ├── dataset_nvda_lstm.csv      # Dataset procesado para LSTM
│   └── nvidia_sentiment_2019_2026.csv  # Datos de sentimiento
├── scripts/
│   ├── lstm/                      # Modelos LSTM
│   │   ├── lstm_model.py
│   │   └── lstm_model_advanced.py
│   ├── random_forest/             # Modelos Random Forest
│   │   ├── random_forest_hyperparameter_optimization.py
│   │   ├── random_forest_hyperparameter_optimization_improved.py
│   │   └── README_OPTIMIZATION.md
│   ├── xgboost/                   # Modelos XGBoost
│   │   ├── xgboost_optuna_optimization.py
│   │   ├── xgboost_optuna_professional.py
│   │   └── README_OPTUNA.md
│   └── preprocessing/             # Preprocesamiento de datos
│       ├── build_dataset.py
│       ├── feature_engineering.py
│       └── merge_datasets.py
├── docs/                          # Documentación
│   └── plan_de_accion.md
├── run_lstm.sh                    # Script para ejecutar LSTM
├── run_rf_improved.sh             # Script para ejecutar Random Forest
├── run_xgboost_professional.sh    # Script para ejecutar XGBoost
└── .env                           # Variables de entorno
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.8+
- pip

### Instalación de Dependencias

```bash
pip install pandas numpy scikit-learn xgboost optuna tensorflow matplotlib seaborn
```

## 💻 Uso

### Ejecutar Modelo LSTM

```bash
chmod +x run_lstm.sh
./run_lstm.sh
```

O en modo silencioso:
```bash
./run_lstm_quiet.sh
```

### Ejecutar Random Forest con Optimización

```bash
chmod +x run_rf_improved.sh
./run_rf_improved.sh
```

### Ejecutar XGBoost con Optuna

```bash
chmod +x run_xgboost_professional.sh
./run_xgboost_professional.sh
```

## 📊 Características

### LSTM
- Arquitectura de red neuronal recurrente
- Procesamiento de secuencias temporales
- Configuración optimizada para evitar mutex locks en macOS

### Random Forest
- Optimización de hiperparámetros con GridSearchCV
- Análisis de importancia de características
- Optimización de umbral de decisión
- Validación cruzada temporal

### XGBoost
- Optimización con Optuna (100 trials)
- Early stopping para prevenir overfitting
- Análisis de importancia de características
- Optimización de umbral de clasificación
- Validación temporal robusta

## 📈 Resultados

Los modelos generan varios archivos de salida:

- **Gráficos de comparación**: Visualización de métricas de rendimiento
- **Importancia de características**: Análisis de features más relevantes
- **Parámetros óptimos**: Configuración de hiperparámetros encontrados
- **Resultados CSV**: Métricas detalladas de evaluación

## 🔧 Configuración

### Variables de Entorno (.env)

El archivo `.env` contiene configuraciones para optimizar el rendimiento en macOS:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TF_NUM_INTRAOP_THREADS=1
TF_NUM_INTEROP_THREADS=1
CUDA_VISIBLE_DEVICES=-1
```

## 📝 Documentación Adicional

- [Plan de Acción](docs/plan_de_accion.md)
- [Optimización Random Forest](scripts/random_forest/README_OPTIMIZATION.md)
- [Optimización XGBoost con Optuna](scripts/xgboost/README_OPTUNA.md)

## 🤝 Contribuciones

Este es un proyecto académico para TFG (Trabajo de Fin de Grado).

## 📄 Licencia

Este proyecto es de uso académico.

## 👤 Autor

Samuel Carrillo - IBM

## 🔗 Enlaces

- Repositorio: [https://github.com/SamuelCarrilloIBM/training_models](https://github.com/SamuelCarrilloIBM/training_models)