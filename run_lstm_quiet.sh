#!/bin/bash
# Script wrapper para ejecutar LSTM sin mostrar warnings de mutex

# Cargar variables de entorno desde .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Variables de entorno cargadas desde .env"
else
    echo "⚠️  Archivo .env no encontrado"
fi

echo "================================"
echo "Configuración de Threading:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "================================"
echo ""
echo "🚀 Iniciando entrenamiento LSTM..."
echo "⏳ Esto puede tardar varios minutos..."
echo ""

# Ejecutar script LSTM filtrando warnings de mutex
python3 scripts/lstm_model.py 2>&1 | grep -v "mutex.cc" | grep -v "RAW: Lock"