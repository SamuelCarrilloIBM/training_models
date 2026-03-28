#!/usr/bin/env python3
"""
Modelo LSTM para predicción de precios de NVIDIA
Utiliza datos financieros y de sentimiento de noticias
"""

# ============================================================================
# CONFIGURACIÓN PARA EVITAR PROBLEMAS DE MUTEX EN macOS
# ============================================================================
import os
os.environ["OMP_NUM_THREADS"] = "2"  # Mejorado: 2 threads en lugar de 1
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configurar matplotlib antes de importar pyplot
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

# Importar TensorFlow con configuraciones especiales
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configurar threading de TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(2)  # Mejorado: 2 threads
tf.config.threading.set_inter_op_parallelism_threads(2)

# Desactivar GPU en Mac de forma segura
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("✅ GPU desactivada, usando CPU")
    except RuntimeError as e:
        print(f"⚠️  No se pudo desactivar GPU: {e}")

# Configurar semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("MODELO LSTM - PREDICCIÓN DE PRECIOS NVIDIA")
print("="*60)

# ============================================================================
# 1. CARGAR Y PREPARAR DATOS
# ============================================================================

print("\n📊 1. Cargando datos...")
df = pd.read_csv('data/dataset_nvda_lstm.csv')
print(f"Dataset cargado: {df.shape}")
print(f"Período: {df['date'].min()} a {df['date'].max()}")

# Separar features y target
exclude_cols = ['date', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y = df['target'].values

print(f"\nFeatures: {len(feature_cols)}")
print(f"Muestras: {len(X)}")
print(f"Distribución del target: {np.bincount(y)}")

# ============================================================================
# 2. DIVIDIR DATOS (TEMPORAL - SIN SHUFFLE)
# ============================================================================

print("\n📈 2. Dividiendo datos temporalmente...")

# 80% train, 20% test (sin shuffle para mantener orden temporal)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Train period: {df['date'].iloc[0]} a {df['date'].iloc[train_size-1]}")
print(f"Test period: {df['date'].iloc[train_size]} a {df['date'].iloc[-1]}")

# ============================================================================
# 3. NORMALIZAR DATOS
# ============================================================================

print("\n🔧 3. Normalizando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4. CREAR SECUENCIAS PARA LSTM
# ============================================================================

def create_sequences(X, y, time_steps=10):
    """
    Crea secuencias temporales para LSTM
    
    Args:
        X: Features
        y: Target
        time_steps: Número de pasos temporales a usar
    
    Returns:
        X_seq: Secuencias de features (samples, time_steps, features)
        y_seq: Targets correspondientes
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    
    return np.array(X_seq), np.array(y_seq)

print("\n🔄 4. Creando secuencias temporales...")
TIME_STEPS = 10  # Usar 10 días de historia

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, TIME_STEPS)

print(f"Train sequences: {X_train_seq.shape}")
print(f"Test sequences: {X_test_seq.shape}")
print(f"Forma de entrada LSTM: (samples={X_train_seq.shape[0]}, time_steps={X_train_seq.shape[1]}, features={X_train_seq.shape[2]})")

# ============================================================================
# 5. CONSTRUIR MODELO LSTM
# ============================================================================

print("\n🏗️  5. Construyendo modelo LSTM (arquitectura simplificada)...")

model = Sequential([
    # Primera capa LSTM (reducida)
    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, X_train_seq.shape[2])),
    Dropout(0.3),
    
    # Segunda capa LSTM
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    
    # Capas densas
    Dense(16, activation='relu'),
    Dropout(0.2),
    
    # Capa de salida (clasificación binaria)
    Dense(1, activation='sigmoid')
])

# Compilar modelo con Focal Loss para clases desbalanceadas
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("✅ Usando Focal Loss (gamma=2.0) para manejar clases desbalanceadas")

print("\n📋 Arquitectura del modelo:")
model.summary()

# ============================================================================
# 6. CALLBACKS (SIMPLIFICADOS PARA EVITAR PROBLEMAS EN macOS)
# ============================================================================

print("\n⚙️  6. Configurando callbacks...")

# Solo Early stopping (simplificado)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

callbacks = [early_stop]
print("✅ Usando solo EarlyStopping para evitar conflictos de threading")

# ============================================================================
# 7. ENTRENAR MODELO
# ============================================================================

print("\n🚀 7. Entrenando modelo LSTM...")
print("="*60)

# Calcular class weights para balancear clases
class_weights = {
    0: len(y_train_seq) / (2 * np.sum(y_train_seq == 0)),
    1: len(y_train_seq) / (2 * np.sum(y_train_seq == 1))
}
print(f"Class weights: {class_weights}")

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=50,  # Reducido para prueba inicial
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

print("\n✅ Entrenamiento completado!")

# ============================================================================
# 8. EVALUAR MODELO
# ============================================================================

print("\n📊 8. Evaluando modelo...")
print("="*60)

# Predicciones
y_train_pred_proba = model.predict(X_train_seq, verbose=0)
y_test_pred_proba = model.predict(X_test_seq, verbose=0)

# Threshold 0.5
y_train_pred = (y_train_pred_proba > 0.5).astype(int).flatten()
y_test_pred = (y_test_pred_proba > 0.5).astype(int).flatten()

# Threshold 0.4 (para mejorar recall)
y_test_pred_04 = (y_test_pred_proba > 0.4).astype(int).flatten()

# Métricas con threshold 0.5
train_acc = accuracy_score(y_train_seq, y_train_pred)
test_acc = accuracy_score(y_test_seq, y_test_pred)
test_f1 = f1_score(y_test_seq, y_test_pred)
test_recall = recall_score(y_test_seq, y_test_pred)
test_precision = precision_score(y_test_seq, y_test_pred)

# Métricas con threshold 0.4
test_acc_04 = accuracy_score(y_test_seq, y_test_pred_04)
test_f1_04 = f1_score(y_test_seq, y_test_pred_04)
test_recall_04 = recall_score(y_test_seq, y_test_pred_04)
test_precision_04 = precision_score(y_test_seq, y_test_pred_04)

print("\n" + "="*60)
print("MÉTRICAS DEL MODELO (Threshold = 0.5)")
print("="*60)
print(f"\nAccuracy en entrenamiento: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Accuracy en prueba: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"\n🎯 Métricas para clase SUBE (1):")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")

print("\n" + "="*60)
print("MÉTRICAS CON THRESHOLD AJUSTADO (0.4)")
print("="*60)
print(f"Accuracy: {test_acc_04:.4f} ({test_acc_04*100:.2f}%)")
print(f"\n🎯 Métricas para clase SUBE (1):")
print(f"  Precision: {test_precision_04:.4f}")
print(f"  Recall:    {test_recall_04:.4f} ⬆️")
print(f"  F1-Score:  {test_f1_04:.4f}")

# Reporte de clasificación
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN (Test Set - Threshold 0.5)")
print("="*60)
print(classification_report(y_test_seq, y_test_pred, target_names=['Baja (0)', 'Sube (1)']))

# Matriz de confusión
cm = confusion_matrix(y_test_seq, y_test_pred)
print("="*60)
print("MATRIZ DE CONFUSIÓN (Test Set)")
print("="*60)
print(f"                Predicho: Baja  Predicho: Sube")
print(f"Real: Baja (0)      {cm[0][0]:6d}          {cm[0][1]:6d}")
print(f"Real: Sube (1)      {cm[1][0]:6d}          {cm[1][1]:6d}")

# ============================================================================
# 9. VISUALIZAR RESULTADOS
# ============================================================================

print("\n📈 9. Generando visualizaciones...")

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Crear figura con 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Loss durante entrenamiento
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Pérdida durante Entrenamiento', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Accuracy durante entrenamiento
axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('Accuracy durante Entrenamiento', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Baja', 'Sube'], yticklabels=['Baja', 'Sube'])
axes[1, 0].set_xlabel('Predicción', fontsize=12)
axes[1, 0].set_ylabel('Real', fontsize=12)
axes[1, 0].set_title('Matriz de Confusión (Test Set)', fontsize=14, fontweight='bold')

# 4. Distribución de probabilidades
axes[1, 1].hist(y_test_pred_proba[y_test_seq == 0], bins=50, alpha=0.6, label='Baja (0)', color='red')
axes[1, 1].hist(y_test_pred_proba[y_test_seq == 1], bins=50, alpha=0.6, label='Sube (1)', color='green')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold 0.5')
axes[1, 1].axvline(x=0.4, color='orange', linestyle='--', linewidth=2, label='Threshold 0.4')
axes[1, 1].set_xlabel('Probabilidad Predicha', fontsize=12)
axes[1, 1].set_ylabel('Frecuencia', fontsize=12)
axes[1, 1].set_title('Distribución de Probabilidades', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_training_results.png', dpi=300, bbox_inches='tight')
print("✅ Gráficos guardados en: lstm_training_results.png")

# ============================================================================
# 10. GUARDAR RESULTADOS
# ============================================================================

print("\n💾 10. Guardando resultados...")

# Guardar métricas
results = {
    'model': 'LSTM',
    'time_steps': TIME_STEPS,
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'test_accuracy_04': test_acc_04,
    'test_precision_04': test_precision_04,
    'test_recall_04': test_recall_04,
    'test_f1_04': test_f1_04
}

results_df = pd.DataFrame([results])
results_df.to_csv('lstm_results.csv', index=False)
print("✅ Métricas guardadas en: lstm_results.csv")

# Guardar modelo
model.save('lstm_model_final.keras')
print("✅ Modelo guardado en: lstm_model_final.keras")

print("\n" + "="*60)
print("✅ ENTRENAMIENTO LSTM COMPLETADO")
print("="*60)
print(f"\n📊 Resumen:")
print(f"  - Accuracy Test: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  - F1-Score: {test_f1:.4f}")
print(f"  - Recall (Sube): {test_recall:.4f}")
print(f"  - Precision (Sube): {test_precision:.4f}")
print(f"\n  Con threshold 0.4:")
print(f"  - Recall (Sube): {test_recall_04:.4f} ⬆️")
print(f"  - F1-Score: {test_f1_04:.4f}")