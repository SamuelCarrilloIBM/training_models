# 🚀 XGBoost con Optimización Optuna

## 📋 Descripción

Este script implementa una optimización avanzada de hiperparámetros para XGBoost utilizando Optuna (optimización Bayesiana), aplicando las mejores prácticas y técnicas de optimización.

## ✅ Mejoras Implementadas

### 1. **Optimización Bayesiana con Optuna** 🔬
- **200 trials** para exploración exhaustiva
- Sampler TPE (Tree-structured Parzen Estimator)
- Aprende de resultados anteriores para converger más rápido

### 2. **Métrica Optimizada: fbeta_score (beta=2)** 📊
```python
fbeta_scorer = make_scorer(fbeta_score, beta=2)
```
- Prioriza **recall** sobre precision
- Mejor para detectar más casos positivos
- Ajustable según necesidades del negocio

### 3. **TimeSeriesSplit con 5 Splits** 🔄
- Validación cruzada temporal robusta
- Respeta el orden cronológico de los datos
- Mayor estabilidad en las métricas

### 4. **Feature Engineering Avanzado** 🎯
Añade ~30 nuevas features automáticamente:
- **Lags**: return_1, return_5, return_10, return_20
- **Rolling stats**: medias móviles, desviaciones, min/max
- **Momentum**: cambios de precio en diferentes ventanas
- **Volatilidad**: desviación estándar de retornos
- **Ratios**: precio relativo a medias móviles
- **Sentimiento**: agregaciones de features de sentimiento

### 5. **Threshold Optimization** 🎯
- 100 puntos de evaluación (0.1 a 0.9)
- Encuentra el threshold óptimo para maximizar F1-Score
- Visualización del impacto del threshold

### 6. **Espacio de Búsqueda Optimizado para XGBoost** ⚙️

```python
params = {
    'booster': ['gbtree', 'dart'],
    'max_depth': [3, 10],
    'min_child_weight': [1, 10],
    'gamma': [0, 0.5],
    'learning_rate': [0.01, 0.3],
    'n_estimators': [100, 500],
    'reg_alpha': [0, 1],
    'reg_lambda': [0, 1],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0]
}
```

## 🚀 Cómo Ejecutar

### Opción 1: Script de ejecución
```bash
chmod +x run_xgboost_optuna.sh
./run_xgboost_optuna.sh
```

### Opción 2: Directamente con Python
```bash
python3 scripts/xgboost/xgboost_optuna_optimization.py
```

## ⏱️ Tiempo de Ejecución

- **Estimado**: 15-25 minutos
- Depende del hardware y número de cores disponibles
- Usa `n_jobs=-1` para paralelización automática

## 📊 Archivos Generados

1. **`xgboost_optuna_results.csv`**
   - Comparación de métricas entre modelos
   - Baseline vs Optuna vs Optuna+Threshold

2. **`xgboost_optuna_comparison.png`**
   - Gráfico comparativo de todas las métricas
   - Accuracy, F1-Score, Precision, Recall

3. **`xgboost_threshold_optimization.png`**
   - Curva de F1-Score vs Threshold
   - Visualización del threshold óptimo

4. **`xgboost_feature_importance.png`**
   - Top 20 features más importantes
   - Basado en gain de XGBoost

5. **`xgboost_best_params.csv`**
   - Mejores hiperparámetros encontrados
   - Listos para usar en producción

## 📈 Mejoras Esperadas

| Métrica | Baseline | Con Optuna | Mejora |
|---------|----------|------------|--------|
| **F1-Score** | ~0.55 | ~0.70-0.75 | +15-20% |
| **Recall** | ~0.60 | ~0.75-0.80 | +15-20% |
| **Accuracy** | ~0.60 | ~0.68-0.72 | +8-12% |

## 🔍 Ventajas de XGBoost con Optuna

### XGBoost
- ✅ Gradient boosting optimizado
- ✅ Manejo nativo de missing values
- ✅ Regularización L1/L2 integrada
- ✅ Early stopping automático
- ✅ Paralelización eficiente
- ✅ Feature importance precisa

### Optuna
- ✅ Optimización Bayesiana inteligente
- ✅ Aprende de trials anteriores
- ✅ Converge más rápido que Grid/Random Search
- ✅ Pruning automático de trials malos
- ✅ Visualizaciones integradas
- ✅ Fácil de usar y extender

## 🎯 Casos de Uso

### 1. Trading Algorítmico
- Priorizar recall para no perder oportunidades
- Threshold ajustable según tolerancia al riesgo

### 2. Detección de Anomalías
- Alta sensibilidad (recall alto)
- Feature importance para interpretabilidad

### 3. Predicción de Series Temporales
- TimeSeriesSplit respeta orden temporal
- Feature engineering captura patrones temporales

## 🔧 Personalización

### Cambiar Beta (prioridad recall vs precision)
```python
# Beta = 2: prioriza recall (detectar más positivos)
fbeta_scorer = make_scorer(fbeta_score, beta=2)

# Beta = 0.5: prioriza precision (menos falsos positivos)
fbeta_scorer = make_scorer(fbeta_score, beta=0.5)

# Beta = 1: balance (equivalente a F1)
fbeta_scorer = make_scorer(fbeta_score, beta=1)
```

### Ajustar Número de Trials
```python
N_TRIALS_OPTUNA = 200  # Más trials = mejor optimización (pero más tiempo)
```

### Modificar Espacio de Búsqueda
```python
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 15),