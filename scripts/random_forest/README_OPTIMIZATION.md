# 🚀 Optimización de Hiperparámetros - Guía Completa

## 📋 Descripción

Este script implementa **3 estrategias avanzadas** de optimización de hiperparámetros para Random Forest, ideal para un TFG de nivel sobresaliente.

## 🎯 Estrategias Implementadas

### 1. **Grid Search** (Búsqueda Exhaustiva)
- ✅ Explora todas las combinaciones posibles
- ⚠️ Muy costoso computacionalmente
- 📊 Garantiza encontrar el mejor en el espacio definido

### 2. **Random Search** (Búsqueda Aleatoria)
- ✅ Más eficiente que Grid Search
- ✅ Explora mejor el espacio de búsqueda
- ✅ Suele encontrar mejores resultados
- 🎲 Recomendado para la mayoría de casos

### 3. **Optimización Bayesiana con Optuna** (Nivel Avanzado)
- 🔥 Aprende de resultados anteriores
- 🔥 Converge rápido a buenos valores
- 🔥 Muy eficiente
- 🎓 Nivel TFG Sobresaliente+

## 📦 Instalación de Dependencias

```bash
# Dependencias básicas (ya deberías tenerlas)
pip install pandas numpy scikit-learn matplotlib seaborn

# Para usar Optuna (RECOMENDADO)
pip install optuna
```

## 🚀 Cómo Ejecutar

### Opción 1: Ejecución Completa (Recomendado)

```bash
cd /Users/samuel/Desktop/Documentación\ TFG/training_models
python3 scripts/random_forest/random