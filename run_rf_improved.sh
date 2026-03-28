#!/bin/bash

echo "🚀 Ejecutando Random Forest Optimizado (Versión Mejorada)"
echo "=========================================================="
echo ""
echo "📋 Mejoras implementadas:"
echo "  ✅ Sin StandardScaler (RF no lo necesita)"
echo "  ✅ Métrica: fbeta_score (beta=2, prioriza recall)"
echo "  ✅ TimeSeriesSplit: 5 splits"
echo "  ✅ Feature engineering avanzado"
echo "  ✅ Threshold optimization: 100 puntos"
echo "  ✅ Optuna: 200 trials"
echo ""
echo "⏱️  Tiempo estimado: 10-15 minutos"
echo ""

python3 scripts/random_forest/random_forest_hyperparameter_optimization_improved.py

echo ""
echo "✅ Ejecución completada"
echo ""
echo "📊 Archivos generados:"
echo "  - hyperparameter_optimization_improved_results.csv"
echo "  - hyperparameter_optimization_improved_comparison.png"
echo "  - threshold_optimization_improved.png"