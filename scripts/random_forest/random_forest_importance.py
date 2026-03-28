#!/usr/bin/env python3
"""
Script para calcular la importancia de características usando Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """
    Carga y prepara los datos para el análisis
    """
    print(f"Cargando datos desde {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Forma del dataset: {df.shape}")
    print(f"\nColumnas disponibles: {list(df.columns)}")
    
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\nValores nulos encontrados:")
        print(null_counts[null_counts > 0])
        # Eliminar filas con valores nulos
        df = df.dropna()
        print(f"Forma después de eliminar nulos: {df.shape}")
    
    return df

def select_features(df):
    """
    Selecciona las características para el modelo
    """
    # Columnas a excluir del análisis
    exclude_cols = ['Date', 'date', 'Unnamed: 0', 'target']
    
    # Identificar la columna objetivo (target = precio del día siguiente)
    target_col = 'target'
    
    # Seleccionar características (todas excepto target y date)
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols]
    
    print(f"\nCaracterísticas seleccionadas ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    print(f"\nTarget: {target_col} (precio del día siguiente)")
    
    return feature_cols, target_col

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, random_state=42):
    """
    Entrena un modelo Random Forest Classifier con regularización para evitar overfitting
    """
    print(f"\nEntrenando Random Forest Classifier con {n_estimators} árboles...")
    print("Hiperparámetros para reducir overfitting:")
    print("  - max_depth: 5 (árboles más simples)")
    print("  - min_samples_split: 20 (más muestras para dividir)")
    print("  - min_samples_leaf: 10 (más muestras por hoja)")
    print("  - max_features: 'sqrt' (menos features por árbol)")
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,              # Reducido de 10 a 5
        min_samples_split=20,     # Aumentado de 5 a 20
        min_samples_leaf=10,      # Aumentado de 2 a 10
        max_features='sqrt',      # Añadido para más diversidad
        class_weight='balanced',  # 🔥 CLAVE: Balancear clases desbalanceadas
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predicciones con threshold por defecto (0.5)
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Predicciones con threshold ajustado (0.4) para mejorar recall de clase 1
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]
    y_test_pred_04 = (y_test_proba > 0.4).astype(int)
    
    # Calcular métricas
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
    test_recall = recall_score(y_test, y_test_pred, pos_label=1)
    test_precision = precision_score(y_test, y_test_pred, pos_label=1)
    
    # Métricas con threshold 0.4
    test_acc_04 = accuracy_score(y_test, y_test_pred_04)
    test_f1_04 = f1_score(y_test, y_test_pred_04, pos_label=1)
    test_recall_04 = recall_score(y_test, y_test_pred_04, pos_label=1)
    test_precision_04 = precision_score(y_test, y_test_pred_04, pos_label=1)
    
    print(f"\n{'='*60}")
    print("MÉTRICAS DEL MODELO (Threshold = 0.5)")
    print(f"{'='*60}")
    print(f"\nAccuracy en entrenamiento: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Accuracy en prueba: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\n🎯 Métricas para clase SUBE (1):")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    print(f"\n{'='*60}")
    print("MÉTRICAS CON THRESHOLD AJUSTADO (0.4)")
    print(f"{'='*60}")
    print(f"Accuracy: {test_acc_04:.4f} ({test_acc_04*100:.2f}%)")
    print(f"\n🎯 Métricas para clase SUBE (1):")
    print(f"  Precision: {test_precision_04:.4f}")
    print(f"  Recall:    {test_recall_04:.4f} ⬆️")
    print(f"  F1-Score:  {test_f1_04:.4f}")
    
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
    
    return rf_model

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

def plot_feature_importance(importance_df, top_n=20, save_path='feature_importance.png'):
    """
    Crea visualizaciones de la importancia de características
    """
    print(f"\nGenerando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Crear figura con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top N características
    top_features = importance_df.head(top_n)
    
    # Gráfico de barras
    axes[0].barh(range(len(top_features)), top_features['Importance'])
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['Feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importancia', fontsize=12)
    axes[0].set_title(f'Top {top_n} Características Más Importantes', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Gráfico de importancia acumulada
    cumsum = top_features['Importance'].cumsum()
    axes[1].plot(range(1, len(top_features) + 1), cumsum, marker='o', linewidth=2, markersize=6)
    axes[1].fill_between(range(1, len(top_features) + 1), cumsum, alpha=0.3)
    axes[1].set_xlabel('Número de Características', fontsize=12)
    axes[1].set_ylabel('Importancia Acumulada', fontsize=12)
    axes[1].set_title('Importancia Acumulada de Características', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% de importancia')
    axes[1].axhline(y=0.9, color='orange', linestyle='--', label='90% de importancia')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {save_path}")
    
    # Análisis de importancia acumulada
    print(f"\n{'='*60}")
    print("ANÁLISIS DE IMPORTANCIA ACUMULADA")
    print(f"{'='*60}")
    
    for threshold in [0.5, 0.8, 0.9, 0.95]:
        n_features = (cumsum <= threshold).sum() + 1
        print(f"Características necesarias para {threshold*100:.0f}% de importancia: {n_features}")
    
    plt.close()

def save_results(importance_df, output_path='feature_importance_results.csv'):
    """
    Guarda los resultados en un archivo CSV
    """
    importance_df.to_csv(output_path, index=False)
    print(f"\nResultados guardados en: {output_path}")

def main():
    """
    Función principal
    """
    print("="*60)
    print("ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS - RANDOM FOREST")
    print("="*60)
    
    # Configuración
    DATA_PATH = 'data/dataset_nvda_lstm.csv'
    OUTPUT_CSV = 'feature_importance_results.csv'
    OUTPUT_PLOT = 'feature_importance_plot.png'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_ESTIMATORS = 100
    TOP_N = 20
    
    try:
        # 1. Cargar datos
        df = load_and_prepare_data(DATA_PATH)
        
        # 2. Seleccionar características
        feature_cols, target_col = select_features(df)
        
        if len(feature_cols) == 0:
            raise ValueError("No se encontraron características válidas")
        
        # 3. Preparar X e y
        X = df[feature_cols].values
        y = df[target_col].values
        
        print(f"\nForma de X: {X.shape}")
        print(f"Forma de y: {y.shape}")
        
        # 4. Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
        )
        
        print(f"\nDatos de entrenamiento: {X_train.shape}")
        print(f"Datos de prueba: {X_test.shape}")
        
        # 5. Escalar características
        print("\nEscalando características...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. Entrenar modelo y evaluar
        model = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test,
                                   n_estimators=N_ESTIMATORS, 
                                   random_state=RANDOM_STATE)
        
        # 8. Analizar importancia
        importance_df = analyze_feature_importance(model, feature_cols, top_n=TOP_N)
        
        # 9. Visualizar
        plot_feature_importance(importance_df, top_n=TOP_N, save_path=OUTPUT_PLOT)
        
        # 10. Guardar resultados
        save_results(importance_df, output_path=OUTPUT_CSV)
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\nError: No se encontró el archivo {DATA_PATH}")
        print("Asegúrate de que el archivo existe en la ruta especificada.")
    except Exception as e:
        print(f"\nError durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()