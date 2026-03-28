#!/bin/bash

echo "🚀 Ejecutando XGBoost con Optimización Optuna"
echo "=============================================="
echo ""
echo "📋 Características:"
echo "  ✅ Optimización Bayesiana (200 trials)"
echo "  ✅ Métrica: fbeta_score (beta=2)"
echo "  ✅ TimeSeriesSplit: 5 splits"
echo "  ✅ Feature engineering avanzado"
echo "  ✅ Threshold optimization"
echo "  ✅ Feature importance analysis"
echo ""
echo "⏱️  Tiempo estimado: 15-25 minutos"
echo ""

python3 scripts/xgboost/xgboost_optuna_optimization.py

echo ""
echo "✅ Ejecución completada"
echo ""
echo "📊 Archivos generados:"
echo "  - xgboost_optuna_results.csv"
echo "  - xgboost_optuna_comparison.png"
echo "  - xgboost_threshold_optimization.png"
echo "  - xgboost_feature_importance.png"
echo "  - xgboost_best_params.csv"