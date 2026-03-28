#!/bin/bash
# Script wrapper para ejecutar LSTM con configuración de entorno correcta

# Cargar variables de entorno desde .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Variables de entorno cargadas desde .env"
else
    echo "⚠️  Archivo .env no encontrado, usando configuración por defecto"
fi

# Mostrar configuración
echo "================================"
echo "Configuración de Threading:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  TF_NUM_INTRAOP_THREADS=$TF_NUM_INTRAOP_THREADS"
echo "  TF_NUM_INTEROP_THREADS=$TF_NUM_INTEROP_THREADS"
echo "================================"
echo ""

# Ejecutar script LSTM
python3 scripts/lstm_model.py