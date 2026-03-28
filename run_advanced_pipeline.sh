#!/bin/bash

echo "=========================================="
echo "PIPELINE AVANZADO DE ENTRENAMIENTO"
echo "=========================================="
echo ""

# 1. Feature Engineering Avanzado
echo "📊 Paso 1: Feature Engineering Avanzado"
echo "=========================================="
python3 scripts/feature_engineering_advanced.py
if [ $? -ne 0 ]; then
    echo "❌ Error en feature engineering"
    exit 1
fi
echo ""

# 2. Entrenamiento LSTM Avanzado
echo "🚀 Paso 2: Entrenamiento LSTM Avanzado"
echo "=========================================="
python3 scripts/lstm_model_advanced.py
if [ $? -ne 0 ]; then
    echo "❌ Error en entrenamiento"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ PIPELINE COMPLETADO"
echo "=========================================="
echo ""
echo "📁 Archivos generados:"
echo "  - data/dataset_nvda_advanced.csv"
echo "  - lstm_advanced_results.png"
echo "  - lstm_advanced_metrics.csv"
echo "  - model_comparison.csv"
echo "  - lstm_model_advanced.keras"
echo ""