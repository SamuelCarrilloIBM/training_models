#!/bin/bash

echo "🚀 XGBoost - Versión Profesional TFG"
echo "====================================="
echo ""
echo "📋 Metodología rigurosa implementada:"
echo "  ✅ Split correcto: Train / Validation / Test"
echo "  ✅ Test set intocable hasta el final"
echo "  ✅ Early stopping en XGBoost"
echo "  ✅ Métrica consistente: fbeta_score (beta=2)"
echo "  ✅ Sin reentrenamiento innecesario"
echo "  ✅ Sin data leakage"
echo ""
echo "⏱️  Tiempo estimado: 10-15 minutos"
echo ""

python3 scripts/xgboost/xgboost_optuna_professional.py

echo ""
echo "✅ Ejecución completada"
echo ""
echo "📊 Archivos generados:"
echo "  - xgboost_professional_results.csv"
echo "  - xgboost_professional_params.csv"
echo "  - xgboost_professional_threshold.png"