#!/usr/bin/env python3
"""
Script para entrenar XGBoost Classifier con filtrado de ruido
Basado en el script de Random Forest pero usando XGBoost
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath, threshold=0.0):
    """
    Carga y prepara los datos SIN filtrar (condiciones reales)
    
    Args:
        filepath: Ruta al archivo CSV
        threshold: Umbral para clasificación (default 0.0 = cualquier subida)
    """
    print(f"📂 Cargando datos desde {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Forma del dataset original: {df.shape}")
    
    # Verificar que existe la columna Close
    if 'Close' not in df.columns:
        raise ValueError("El dataset debe contener la columna 'Close'")
    
    # Calcular retornos futuros (t+1)
    returns = df['Close'].pct_change().shift(-1)
    
    # Target binario simple: 1 si sube, 0 si baja
    y = np.where(returns > threshold, 1, 0)
    
    # Eliminar solo NaN (última fila)
    mask = ~np.isnan(returns)
    df_clean = df[mask].copy()
    y_clean = y[mask]
    
    print(f"\n📊 Preparación de datos:")
    print(f"   Threshold: {threshold} (cualquier movimiento)")
    print(f"   Muestras totales: {len(df_clean)}")
    print(f"   Distribución del target:")
    print(f"     Sube (1): {np.sum(y_clean == 1)} ({np.sum(y_clean == 1) / len(y_clean) * 100:.1f}%)")
    print(f"     Baja (0): {np.sum(y_clean == 0)} ({np.sum(y_clean == 0) / len(y_clean) * 100:.1f}%)")
    
    return df_clean, y_clean

def select_features(df):
    """
    Selecciona las características para el modelo
    """
    # Columnas a excluir del análisis
    exclude_cols = ['Date', 'date', 'Unnamed: 0', 'target', 'Close', 'High', 'Low', 'Volume']
    
    # Seleccionar características
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols]
    
    print(f"\n📋 Características seleccionadas ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    return feature_cols

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo XGBoost Classifier con regularización
    """
    print(f"\n🚀 Entrenando XGBoost Classifier...")
    print("Hiperparámetros:")
    print("  - max_depth: 5")
    print("  - learning_rate: 0.1")
    print("  - n_estimators: 100")
    print("  - min_child_weight: 3")
    print("  - gamma: 0.1")
    print("  - subsample: 0.8")
    print("  - colsample_bytree: 0.8")
    print("  - scale_pos_weight: auto (para balancear clases)")
    
    # Calcular scale_pos_weight para balancear clases
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos
    
    print(f"  - scale_pos_weight calculado: {scale_pos_weight:.2f}")
    
    xgb_model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # Entrenar con early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predicciones
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    # Predicciones con probabilidades
    y_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
    test_recall = recall_score(y_test, y_test_pred, pos_label=1)
    test_precision = precision_score(y_test, y_test_pred, pos_label=1)
    
    print(f"\n{'='*60}")
    print("MÉTRICAS DEL MODELO")
    print(f"{'='*60}")
    print(f"\nAccuracy en entrenamiento: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Accuracy en prueba: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\n🎯 Métricas para clase SUBE (1):")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    # Reporte de clasificación
    print(f"\n{'='*60}")
    print("REPORTE DE CLASIFICACIÓN (Test Set)")
    print(f"{'='*60}")
    print(classification_report(y_test, y_test_pred, target_names=['Baja (0)', 'Sube (1)']))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n{'='*60}")
    print("MATRIZ DE CONFUSIÓN (Test Set)")
    print(f"{'='*60}")
    print(f"                Predicho: Baja  Predicho: Sube")
    print(f"Real: Baja (0)      {cm[0][0]:6d}          {cm[0][1]:6d}")
    print(f"Real: Sube (1)      {cm[1][0]:6d}          {cm[1][1]:6d}")
    
    return xgb_model, y_test_proba

def analyze_feature_importance(model, feature_names, top_n=20):
    """
    Analiza y visualiza la importancia de características
    """
    # Obtener importancias
    importances = model.feature_importances_
    
    # Crear DataFrame con importancias
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"TOP {top_n} CARACTERÍSTICAS MÁS IMPORTANTES")
    print(f"{'='*60}")
    
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"{row['Feature']:40s} {row['Importance']:.6f}")
    
    return importance_df

def plot_results(importance_df, y_test, y_test_proba, top_n=20, save_path='xgboost_results.png'):
    """
    Crea visualizaciones de resultados
    """
    print(f"\n📊 Generando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Crear figura con tres subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Top N características
    top_features = importance_df.head(top_n)
    axes[0].barh(range(len(top_features)), top_features['Importance'])
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['Feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importancia', fontsize=12)
    axes[0].set_title(f'Top {top_n} Características Más Importantes', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # 2. Importancia acumulada
    cumsum = top_features['Importance'].cumsum()
    axes[1].plot(range(1, len(top_features) + 1), cumsum, marker='o', linewidth=2, markersize=6)
    axes[1].fill_between(range(1, len(top_features) + 1), cumsum, alpha=0.3)
    axes[1].set_xlabel('Número de Características', fontsize=12)
    axes[1].set_ylabel('Importancia Acumulada', fontsize=12)
    axes[1].set_title('Importancia Acumulada', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.8, color='r', linestyle='--', label='80%')
    axes[1].legend()
    
    # 3. Distribución de probabilidades
    y_pred = (y_test_proba > 0.5).astype(int)
    axes[2].hist(y_test_proba[y_test == 0], bins=50, alpha=0.5, label='Baja (0)', edgecolor='black')
    axes[2].hist(y_test_proba[y_test == 1], bins=50, alpha=0.5, label='Sube (1)', edgecolor='black')
    axes[2].axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Umbral')
    axes[2].set_xlabel('Probabilidad Predicha', fontsize=12)
    axes[2].set_ylabel('Frecuencia', fontsize=12)
    axes[2].set_title('Distribución de Probabilidades', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado en: {save_path}")
    plt.close()

def save_results(importance_df, output_path='xgboost_feature_importance.csv'):
    """
    Guarda los resultados en un archivo CSV
    """
    importance_df.to_csv(output_path, index=False)
    print(f"💾 Resultados guardados en: {output_path}")

def main():
    """
    Función principal
    """
    print("="*60)
    print("XGBOOST CLASSIFIER CON FILTRADO DE RUIDO")
    print("="*60)
    
    # Configuración
    DATA_PATH = 'data/dataset_nvda_lstm.csv'
    OUTPUT_CSV = 'xgboost_feature_importance.csv'
    OUTPUT_PLOT = 'xgboost_results.png'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    THRESHOLD = 0.005  # 0.5%
    TOP_N = 20
    
    try:
        # 1. Cargar datos con filtrado de ruido
        df, y = load_and_prepare_data(DATA_PATH, threshold=THRESHOLD)
        
        # 2. Seleccionar características
        feature_cols = select_features(df)
        
        if len(feature_cols) == 0:
            raise ValueError("No se encontraron características válidas")
        
        # 3. Preparar X
        X = df[feature_cols].values
        
        print(f"\nForma de X: {X.shape}")
        print(f"Forma de y: {y.shape}")
        
        # 4. Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
        )
        
        print(f"\nDatos de entrenamiento: {X_train.shape}")
        print(f"Datos de prueba: {X_test.shape}")
        
        # 5. Escalar características
        print("\n⚙️  Escalando características...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. Entrenar modelo
        model, y_test_proba = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # 7. Analizar importancia
        importance_df = analyze_feature_importance(model, feature_cols, top_n=TOP_N)
        
        # 8. Visualizar
        plot_results(importance_df, y_test, y_test_proba, top_n=TOP_N, save_path=OUTPUT_PLOT)
        
        # 9. Guardar resultados
        save_results(importance_df, output_path=OUTPUT_CSV)
        
        print("\n" + "="*60)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró el archivo {DATA_PATH}")
        print("Asegúrate de que el archivo existe en la ruta especificada.")
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()