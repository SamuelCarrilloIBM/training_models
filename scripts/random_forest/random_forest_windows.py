#!/usr/bin/env python3
"""
Script para calcular la importancia de características usando Random Forest
con ventanas temporales (1 día, 1 semana, 1 mes)
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

def create_window_features(df):
    """
    Crea características basadas en ventanas temporales
    
    Ventanas:
    - 1 día: lags individuales (ya existen)
    - 1 semana: rolling 5 días
    - 1 mes: rolling 20 días
    """
    print("\n" + "="*60)
    print("CREANDO CARACTERÍSTICAS DE VENTANAS TEMPORALES")
    print("="*60)
    
    WINDOW_1W = 5   # 1 semana = 5 días de trading
    WINDOW_1M = 20  # 1 mes = ~20 días de trading
    
    df_windowed = df.copy()
    
    # ========================================
    # VENTANA 1 SEMANA (5 días)
    # ========================================
    print(f"\n📊 Creando features de ventana 1 semana ({WINDOW_1W} días)...")
    
    # Retornos
    df_windowed['return_mean_1w'] = df['log_return'].rolling(WINDOW_1W).mean().shift(1)
    df_windowed['return_std_1w'] = df['log_return'].rolling(WINDOW_1W).std().shift(1)
    df_windowed['return_max_1w'] = df['log_return'].rolling(WINDOW_1W).max().shift(1)
    df_windowed['return_min_1w'] = df['log_return'].rolling(WINDOW_1W).min().shift(1)
    
    # Volumen
    df_windowed['volume_mean_1w'] = df['Volume'].rolling(WINDOW_1W).mean().shift(1)
    df_windowed['volume_std_1w'] = df['Volume'].rolling(WINDOW_1W).std().shift(1)
    
    # Volatilidad
    df_windowed['volatility_mean_1w'] = df['volatility_7d'].rolling(WINDOW_1W).mean().shift(1)
    
    # RSI
    df_windowed['rsi_mean_1w'] = df['rsi_14'].rolling(WINDOW_1W).mean().shift(1)
    
    # Momentum (cambio desde inicio de ventana)
    df_windowed['momentum_1w'] = (df['Close'] / df['Close'].shift(WINDOW_1W) - 1).shift(1)
    
    # Sentimiento
    if 'mean_tone_shifted' in df.columns:
        df_windowed['sentiment_mean_1w'] = df['mean_tone_shifted'].rolling(WINDOW_1W).mean().shift(1)
        df_windowed['sentiment_std_1w'] = df['mean_tone_shifted'].rolling(WINDOW_1W).std().shift(1)
    
    if 'n_news_shifted' in df.columns:
        df_windowed['news_sum_1w'] = df['n_news_shifted'].rolling(WINDOW_1W).sum().shift(1)
    
    print(f"  ✓ Creadas {11} features de 1 semana")
    
    # ========================================
    # VENTANA 1 MES (20 días)
    # ========================================
    print(f"\n📊 Creando features de ventana 1 mes ({WINDOW_1M} días)...")
    
    # Retornos
    df_windowed['return_mean_1m'] = df['log_return'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['return_std_1m'] = df['log_return'].rolling(WINDOW_1M).std().shift(1)
    df_windowed['return_max_1m'] = df['log_return'].rolling(WINDOW_1M).max().shift(1)
    df_windowed['return_min_1m'] = df['log_return'].rolling(WINDOW_1M).min().shift(1)
    
    # Volumen
    df_windowed['volume_mean_1m'] = df['Volume'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['volume_std_1m'] = df['Volume'].rolling(WINDOW_1M).std().shift(1)
    
    # Volatilidad
    df_windowed['volatility_mean_1m'] = df['volatility_7d'].rolling(WINDOW_1M).mean().shift(1)
    
    # RSI
    df_windowed['rsi_mean_1m'] = df['rsi_14'].rolling(WINDOW_1M).mean().shift(1)
    
    # Momentum (cambio desde inicio de ventana)
    df_windowed['momentum_1m'] = (df['Close'] / df['Close'].shift(WINDOW_1M) - 1).shift(1)
    
    # Sentimiento
    if 'mean_tone_shifted' in df.columns:
        df_windowed['sentiment_mean_1m'] = df['mean_tone_shifted'].rolling(WINDOW_1M).mean().shift(1)
        df_windowed['sentiment_std_1m'] = df['mean_tone_shifted'].rolling(WINDOW_1M).std().shift(1)
    
    if 'n_news_shifted' in df.columns:
        df_windowed['news_sum_1m'] = df['n_news_shifted'].rolling(WINDOW_1M).sum().shift(1)
    
    print(f"  ✓ Creadas {11} features de 1 mes")
    
    # ========================================
    # FEATURES ADICIONALES: Ratios entre ventanas
    # ========================================
    print(f"\n📊 Creando features de ratios entre ventanas...")
    
    # Ratio volatilidad 1w vs 1m (detecta cambios de régimen)
    df_windowed['volatility_ratio_1w_1m'] = (
        df_windowed['return_std_1w'] / df_windowed['return_std_1m']
    )
    
    # Ratio volumen 1w vs 1m (detecta cambios de interés)
    df_windowed['volume_ratio_1w_1m'] = (
        df_windowed['volume_mean_1w'] / df_windowed['volume_mean_1m']
    )
    
    # Diferencia momentum (aceleración/desaceleración)
    df_windowed['momentum_diff_1w_1m'] = (
        df_windowed['momentum_1w'] - df_windowed['momentum_1m']
    )
    
    print(f"  ✓ Creadas {3} features de ratios")
    
    # ========================================
    # RESUMEN
    # ========================================
    print(f"\n{'='*60}")
    print("RESUMEN DE FEATURES CREADAS")
    print(f"{'='*60}")
    print(f"Features originales: {len(df.columns)}")
    print(f"Features con ventanas: {len(df_windowed.columns)}")
    print(f"Nuevas features añadidas: {len(df_windowed.columns) - len(df.columns)}")
    
    # Eliminar filas con NaN (debido a rolling windows)
    rows_before = len(df_windowed)
    df_windowed = df_windowed.dropna()
    rows_after = len(df_windowed)
    print(f"\nFilas eliminadas por NaN: {rows_before - rows_after}")
    print(f"Filas finales: {rows_after}")
    
    return df_windowed

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
    
    # Identificar la columna objetivo
    target_col = 'target'
    
    # Seleccionar características (todas excepto target y date)
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols]
    
    print(f"\n{'='*60}")
    print(f"CARACTERÍSTICAS SELECCIONADAS ({len(feature_cols)})")
    print(f"{'='*60}")
    
    # Agrupar por tipo
    original_features = [col for col in feature_cols if not any(x in col for x in ['_1w', '_1m', 'lag_'])]
    lag_features = [col for col in feature_cols if 'lag_' in col]
    window_1w_features = [col for col in feature_cols if '_1w' in col]
    window_1m_features = [col for col in feature_cols if '_1m' in col]
    
    print(f"\n📊 Features originales ({len(original_features)}):")
    for col in original_features[:10]:
        print(f"  • {col}")
    if len(original_features) > 10:
        print(f"  ... y {len(original_features) - 10} más")
    
    print(f"\n📅 Features de lags ({len(lag_features)}):")
    for col in lag_features:
        print(f"  • {col}")
    
    print(f"\n📈 Features de ventana 1 semana ({len(window_1w_features)}):")
    for col in window_1w_features:
        print(f"  • {col}")
    
    print(f"\n📈 Features de ventana 1 mes ({len(window_1m_features)}):")
    for col in window_1m_features:
        print(f"  • {col}")
    
    print(f"\nTarget: {target_col}")
    
    return feature_cols, target_col

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, random_state=42):
    """
    Entrena un modelo Random Forest Classifier
    """
    print(f"\n{'='*60}")
    print(f"ENTRENANDO RANDOM FOREST CON {n_estimators} ÁRBOLES")
    print(f"{'='*60}")
    print("\nHiperparámetros:")
    print("  • max_depth: 5")
    print("  • min_samples_split: 20")
    print("  • min_samples_leaf: 10")
    print("  • max_features: 'sqrt'")
    print("  • class_weight: 'balanced'")
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
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
    
    return rf_model

def analyze_feature_importance(model, feature_names, top_n=30):
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
        # Identificar tipo de feature
        if '_1w' in row['Feature']:
            tipo = "🔵 1W"
        elif '_1m' in row['Feature']:
            tipo = "🟢 1M"
        elif 'lag_' in row['Feature']:
            tipo = "🟡 LAG"
        else:
            tipo = "⚪ BASE"
        
        print(f"{tipo} {row['Feature']:45s} {row['Importance']:.6f}")
    
    # Análisis por tipo de feature
    print(f"\n{'='*60}")
    print("ANÁLISIS POR TIPO DE FEATURE")
    print(f"{'='*60}")
    
    base_features = importance_df[~importance_df['Feature'].str.contains('_1w|_1m|lag_')]
    lag_features = importance_df[importance_df['Feature'].str.contains('lag_')]
    window_1w = importance_df[importance_df['Feature'].str.contains('_1w')]
    window_1m = importance_df[importance_df['Feature'].str.contains('_1m')]
    
    print(f"\n⚪ Features BASE:")
    print(f"   Total: {len(base_features)}")
    print(f"   Importancia promedio: {base_features['Importance'].mean():.6f}")
    print(f"   Importancia total: {base_features['Importance'].sum():.6f}")
    
    print(f"\n🟡 Features LAG:")
    print(f"   Total: {len(lag_features)}")
    print(f"   Importancia promedio: {lag_features['Importance'].mean():.6f}")
    print(f"   Importancia total: {lag_features['Importance'].sum():.6f}")
    
    print(f"\n🔵 Features VENTANA 1 SEMANA:")
    print(f"   Total: {len(window_1w)}")
    print(f"   Importancia promedio: {window_1w['Importance'].mean():.6f}")
    print(f"   Importancia total: {window_1w['Importance'].sum():.6f}")
    
    print(f"\n🟢 Features VENTANA 1 MES:")
    print(f"   Total: {len(window_1m)}")
    print(f"   Importancia promedio: {window_1m['Importance'].mean():.6f}")
    print(f"   Importancia total: {window_1m['Importance'].sum():.6f}")
    
    return importance_df

def plot_feature_importance(importance_df, top_n=30, save_path='feature_importance_windows.png'):
    """
    Crea visualizaciones de la importancia de características
    """
    print(f"\nGenerando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Crear figura con tres subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    # Top N características
    top_features = importance_df.head(top_n)
    
    # Colores por tipo
    colors = []
    for feat in top_features['Feature']:
        if '_1w' in feat:
            colors.append('#3498db')  # Azul
        elif '_1m' in feat:
            colors.append('#2ecc71')  # Verde
        elif 'lag_' in feat:
            colors.append('#f39c12')  # Naranja
        else:
            colors.append('#95a5a6')  # Gris
    
    # Gráfico 1: Barras horizontales
    axes[0].barh(range(len(top_features)), top_features['Importance'], color=colors)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['Feature'], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importancia', fontsize=12)
    axes[0].set_title(f'Top {top_n} Características Más Importantes', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#95a5a6', label='Base'),
        Patch(facecolor='#f39c12', label='Lag'),
        Patch(facecolor='#3498db', label='1 Semana'),
        Patch(facecolor='#2ecc71', label='1 Mes')
    ]
    axes[0].legend(handles=legend_elements, loc='lower right')
    
    # Gráfico 2: Importancia acumulada
    cumsum = top_features['Importance'].cumsum()
    axes[1].plot(range(1, len(top_features) + 1), cumsum, marker='o', linewidth=2, markersize=4)
    axes[1].fill_between(range(1, len(top_features) + 1), cumsum, alpha=0.3)
    axes[1].set_xlabel('Número de Características', fontsize=12)
    axes[1].set_ylabel('Importancia Acumulada', fontsize=12)
    axes[1].set_title('Importancia Acumulada', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.8, color='r', linestyle='--', label='80%')
    axes[1].axhline(y=0.9, color='orange', linestyle='--', label='90%')
    axes[1].legend()
    
    # Gráfico 3: Importancia por tipo
    base_features = importance_df[~importance_df['Feature'].str.contains('_1w|_1m|lag_')]
    lag_features = importance_df[importance_df['Feature'].str.contains('lag_')]
    window_1w = importance_df[importance_df['Feature'].str.contains('_1w')]
    window_1m = importance_df[importance_df['Feature'].str.contains('_1m')]
    
    types = ['Base', 'Lag', '1 Semana', '1 Mes']
    importances = [
        base_features['Importance'].sum(),
        lag_features['Importance'].sum(),
        window_1w['Importance'].sum(),
        window_1m['Importance'].sum()
    ]
    colors_pie = ['#95a5a6', '#f39c12', '#3498db', '#2ecc71']
    
    axes[2].pie(importances, labels=types, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    axes[2].set_title('Importancia Total por Tipo de Feature', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {save_path}")
    
    plt.close()

def save_results(importance_df, output_path='feature_importance_windows.csv'):
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
    print("RANDOM FOREST CON VENTANAS TEMPORALES")
    print("="*60)
    
    # Configuración
    DATA_PATH = 'data/dataset_nvda_lstm.csv'
    OUTPUT_CSV = 'feature_importance_windows.csv'
    OUTPUT_PLOT = 'feature_importance_windows.png'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_ESTIMATORS = 100
    TOP_N = 30
    
    try:
        # 1. Cargar datos
        df = load_and_prepare_data(DATA_PATH)
        
        # 2. Crear features de ventanas temporales
        df_windowed = create_window_features(df)
        
        # 3. Seleccionar características
        feature_cols, target_col = select_features(df_windowed)
        
        if len(feature_cols) == 0:
            raise ValueError("No se encontraron características válidas")
        
        # 4. Preparar X e y
        X = df_windowed[feature_cols].values
        y = df_windowed[target_col].values
        
        print(f"\n{'='*60}")
        print("PREPARACIÓN DE DATOS")
        print(f"{'='*60}")
        print(f"Forma de X: {X.shape}")
        print(f"Forma de y: {y.shape}")
        
        # 5. Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
        )
        
        print(f"\nDatos de entrenamiento: {X_train.shape}")
        print(f"Datos de prueba: {X_test.shape}")
        
        # 6. Escalar características
        print("\nEscalando características...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 7. Entrenar modelo
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