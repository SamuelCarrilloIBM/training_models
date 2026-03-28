#!/usr/bin/env python3
"""
Modelo LSTM Avanzado para Predicción de Precios NVDA
Incluye arquitectura mejorada, regularización y comparación con modelos simples
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Configurar semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)


class AdvancedLSTMModel:
    """
    Modelo LSTM avanzado con múltiples arquitecturas y comparación con baselines
    """
    
    def __init__(self, sequence_length: int = 30):
        """
        Inicializa el modelo
        
        Args:
            sequence_length: Longitud de la secuencia temporal
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.feature_names = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close', threshold: float = 0.005):
        """
        Prepara los datos para entrenamiento de clasificación binaria con filtrado de ruido
        
        Args:
            df: DataFrame con features
            target_col: Columna de precio para calcular retornos
            threshold: Umbral para clasificar movimientos significativos (default 0.005 = 0.5%)
        
        Returns:
            Tupla con datos de entrenamiento y validación
        """
        print("📊 Preparando datos para clasificación binaria (filtrando ruido)...")
        
        # Separar features
        feature_cols = [col for col in df.columns if col not in ['date', target_col, 'target']]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        
        # Calcular retornos futuros (t+1)
        returns = df[target_col].pct_change().shift(-1)
        
        # Clasificación en 3 categorías:
        # 1 = Sube fuerte (> threshold)
        # 0 = Baja fuerte (< -threshold)
        # -1 = Lateral (entre -threshold y threshold) -> FILTRAR
        y = np.where(returns > threshold, 1,
                     np.where(returns < -threshold, 0, -1))
        
        # Eliminar NaN
        mask = ~np.isnan(returns)
        X = X[mask]
        y = y[mask]
        returns_clean = returns[mask]
        
        # Filtrar movimientos laterales (ruido)
        significant_mask = y != -1
        X = X[significant_mask]
        y = y[significant_mask]
        returns_clean = returns_clean[significant_mask]
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Umbral de movimiento significativo: ±{threshold*100:.1f}%")
        print(f"   Muestras totales: {len(mask)}")
        print(f"   Muestras con movimiento significativo: {len(X)} ({len(X)/len(mask)*100:.1f}%)")
        print(f"   Muestras filtradas (ruido): {len(mask) - len(X)} ({(len(mask)-len(X))/len(mask)*100:.1f}%)")
        print(f"   Distribución del target:")
        print(f"     Sube fuerte (1): {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
        print(f"     Baja fuerte (0): {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Crear secuencias temporales
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Split train/validation/test
        # 70% train, 15% validation, 15% test
        train_size = int(0.7 * len(X_seq))
        val_size = int(0.15 * len(X_seq))
        
        X_train = X_seq[:train_size]
        y_train = y_seq[:train_size]
        
        X_val = X_seq[train_size:train_size + val_size]
        y_val = y_seq[train_size:train_size + val_size]
        
        X_test = X_seq[train_size + val_size:]
        y_test = y_seq[train_size + val_size:]
        
        print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """
        Crea secuencias temporales para LSTM
        
        Args:
            X: Features normalizadas
            y: Target
        
        Returns:
            Tupla (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, 
                         input_shape: tuple,
                         architecture: str = 'standard',
                         dropout_rate: float = 0.3,
                         l2_reg: float = 0.01):
        """
        Construye el modelo LSTM
        
        Args:
            input_shape: Shape de entrada (sequence_length, n_features)
            architecture: Tipo de arquitectura ('simple', 'standard', 'deep', 'bidirectional')
            dropout_rate: Tasa de dropout
            l2_reg: Regularización L2
        
        Returns:
            Modelo compilado
        """
        print(f"🏗️  Construyendo modelo LSTM ({architecture})...")
        
        model = keras.Sequential()
        
        if architecture == 'simple':
            # Arquitectura simple: 1 capa LSTM
            model.add(layers.LSTM(
                64, 
                input_shape=input_shape,
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(layers.Dropout(dropout_rate))
            
        elif architecture == 'standard':
            # Arquitectura estándar: 2 capas LSTM
            model.add(layers.LSTM(
                64, 
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.LSTM(
                32,
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(layers.Dropout(dropout_rate))
            
        elif architecture == 'deep':
            # Arquitectura profunda: 3 capas LSTM
            model.add(layers.LSTM(
                128, 
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.LSTM(
                64,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.LSTM(
                32,
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(layers.Dropout(dropout_rate))
            
        elif architecture == 'bidirectional':
            # Arquitectura bidireccional
            model.add(layers.Bidirectional(
                layers.LSTM(64, return_sequences=True),
                input_shape=input_shape
            ))
            model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.Bidirectional(
                layers.LSTM(32)
            ))
            model.add(layers.Dropout(dropout_rate))
        
        # Capas densas finales para clasificación binaria
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(dropout_rate / 2))
        model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid para clasificación binaria
        
        # Compilar con binary_crossentropy
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',  # Loss para clasificación binaria
            metrics=['accuracy']  # Accuracy en vez de MAE
        )
        
        print(f"   Parámetros totales: {model.count_params():,}")
        
        return model
    
    def train(self, 
              train_data: tuple,
              val_data: tuple,
              architecture: str = 'standard',
              epochs: int = 100,
              batch_size: int = 32):
        """
        Entrena el modelo LSTM
        
        Args:
            train_data: Tupla (X_train, y_train)
            val_data: Tupla (X_val, y_val)
            architecture: Tipo de arquitectura
            epochs: Número de épocas
            batch_size: Tamaño del batch
        
        Returns:
            Historia del entrenamiento
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Construir modelo
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_lstm_model(input_shape, architecture=architecture)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'lstm_model_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        print(f"🚀 Entrenando modelo...")
        print(f"   Épocas: {epochs} | Batch size: {batch_size}")
        
        # Entrenar
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Entrenamiento completado!")
        
        return self.history
    
    def evaluate(self, test_data: tuple):
        """
        Evalúa el modelo de clasificación en el conjunto de test
        
        Args:
            test_data: Tupla (X_test, y_test)
        
        Returns:
            Diccionario con métricas de clasificación
        """
        X_test, y_test = test_data
        
        print("📊 Evaluando modelo en test set...")
        
        # Predicciones (probabilidades)
        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        
        # Convertir probabilidades a clases (umbral 0.5)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Métricas de clasificación
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        print("\n📈 Métricas de Clasificación en Test Set:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        print("\n📊 Matriz de Confusión:")
        print(f"   TN: {cm[0,0]} | FP: {cm[0,1]}")
        print(f"   FN: {cm[1,0]} | TP: {cm[1,1]}")
        
        print("\n📋 Reporte de Clasificación:")
        print(classification_report(y_test, y_pred, target_names=['Baja (0)', 'Sube (1)']))
        
        return metrics, y_pred_proba
    
    def compare_with_baselines(self, train_data: tuple, test_data: tuple):
        """
        Compara LSTM con modelos baseline de clasificación
        
        Args:
            train_data: Tupla (X_train, y_train)
            test_data: Tupla (X_test, y_test)
        
        Returns:
            DataFrame con comparación de métricas
        """
        print("\n🔬 Comparando con modelos baseline de clasificación...")
        
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # Aplanar secuencias para modelos tradicionales
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        results = []
        
        # 1. Logistic Regression
        print("   Entrenando Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_flat, y_train)
        y_pred_lr = lr.predict(X_test_flat)
        
        results.append({
            'Model': 'Logistic Regression',
            'Accuracy': accuracy_score(y_test, y_pred_lr),
            'Precision': precision_score(y_test, y_pred_lr, zero_division=0),
            'Recall': recall_score(y_test, y_pred_lr, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_lr, zero_division=0)
        })
        
        # 2. Random Forest Classifier
        print("   Entrenando Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_flat, y_train)
        y_pred_rf = rf.predict(X_test_flat)
        
        results.append({
            'Model': 'Random Forest',
            'Accuracy': accuracy_score(y_test, y_pred_rf),
            'Precision': precision_score(y_test, y_pred_rf, zero_division=0),
            'Recall': recall_score(y_test, y_pred_rf, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_rf, zero_division=0)
        })
        
        # 3. LSTM (ya entrenado)
        y_pred_proba_lstm = self.model.predict(X_test, verbose=0).flatten()
        y_pred_lstm = (y_pred_proba_lstm > 0.5).astype(int)
        
        results.append({
            'Model': 'LSTM',
            'Accuracy': accuracy_score(y_test, y_pred_lstm),
            'Precision': precision_score(y_test, y_pred_lstm, zero_division=0),
            'Recall': recall_score(y_test, y_pred_lstm, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_lstm, zero_division=0)
        })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\n📊 Comparación de Modelos:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_results(self, test_data: tuple, y_pred_proba: np.ndarray, save_path: str = None):
        """
        Visualiza los resultados del modelo de clasificación
        
        Args:
            test_data: Tupla (X_test, y_test)
            y_pred_proba: Probabilidades predichas
            save_path: Ruta para guardar la figura
        """
        _, y_test = test_data
        
        # Convertir probabilidades a clases
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Loss durante entrenamiento
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Pérdida durante Entrenamiento', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy durante entrenamiento
        axes[0, 1].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Accuracy durante Entrenamiento', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                    xticklabels=['Baja (0)', 'Sube (1)'],
                    yticklabels=['Baja (0)', 'Sube (1)'])
        axes[1, 0].set_xlabel('Predicción', fontsize=12)
        axes[1, 0].set_ylabel('Real', fontsize=12)
        axes[1, 0].set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        
        # 4. Distribución de probabilidades predichas
        axes[1, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Baja (0)', edgecolor='black')
        axes[1, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Sube (1)', edgecolor='black')
        axes[1, 1].axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Umbral')
        axes[1, 1].set_xlabel('Probabilidad Predicha', fontsize=12)
        axes[1, 1].set_ylabel('Frecuencia', fontsize=12)
        axes[1, 1].set_title('Distribución de Probabilidades', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráficos guardados en: {save_path}")
        
        plt.close()


def main():
    """
    Función principal para ejecutar el entrenamiento y evaluación
    """
    print("="*60)
    print("MODELO LSTM AVANZADO - PREDICCIÓN DE PRECIOS NVIDIA")
    print("="*60)
    
    # Cargar dataset
    print("\n📂 Cargando dataset avanzado...")
    df = pd.read_csv('data/dataset_nvda_lstm.csv', parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Dataset: {df.shape}")
    print(f"Período: {df['date'].min()} a {df['date'].max()}")
    
    # Inicializar modelo
    model = AdvancedLSTMModel(sequence_length=30)
    
    # Preparar datos
    train_data, val_data, test_data = model.prepare_data(df, target_col='Close')
    
    # Entrenar modelo con arquitectura estándar
    print("\n" + "="*60)
    print("ENTRENANDO MODELO LSTM")
    print("="*60)
    
    model.train(
        train_data=train_data,
        val_data=val_data,
        architecture='standard',
        epochs=100,
        batch_size=32
    )
    
    # Evaluar modelo
    print("\n" + "="*60)
    print("EVALUACIÓN DEL MODELO")
    print("="*60)
    
    metrics, y_pred = model.evaluate(test_data)
    
    # Comparar con baselines
    comparison_df = model.compare_with_baselines(train_data, test_data)
    
    # Visualizar resultados
    print("\n📊 Generando visualizaciones...")
    model.plot_results(test_data, y_pred, save_path='lstm_advanced_results.png')
    
    # Guardar resultados
    print("\n💾 Guardando resultados...")
    
    # Guardar métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('lstm_advanced_metrics.csv', index=False)
    
    # Guardar comparación
    comparison_df.to_csv('model_comparison.csv', index=False)
    
    # Guardar modelo
    model.model.save('lstm_model_advanced.keras')
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\n📊 Resumen de Resultados:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    
    print(f"\n📁 Archivos generados:")
    print(f"  - lstm_advanced_results.png")
    print(f"  - lstm_advanced_metrics.csv")
    print(f"  - model_comparison.csv")
    print(f"  - lstm_model_advanced.keras")
    
    return model, metrics, comparison_df


if __name__ == "__main__":
    # Configurar para evitar problemas en macOS
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Configurar matplotlib
    import matplotlib
    matplotlib.use('Agg')
    
    # Ejecutar
    model, metrics, comparison = main()
