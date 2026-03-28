#!/usr/bin/env python3
"""
MULTI-HORIZON PREDICTION - Random Forest
Entrena modelos para predecir movimientos de precio en diferentes horizontes temporales:
- 1 día (t+1)
- 1 semana (t+5)
- 1 mes (t+20)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def create_multi_horizon_targets(df):
    """
    Crea targets para diferentes horizontes temporales
    """
    print("\n" + "="*80)
    print("CREANDO TARGETS PARA MÚLTIPLES HORIZONTES TEMPORALES")
    print("="*80)
    
    df_targets = df.copy()
    
    # Target 1 DÍA: precio[t+1] > precio[t]
    df_targets['target_1d'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    print(f"\n📅 Target 1 DÍA (t+1):")
    print(f"   Fórmula: precio[t+1] > precio[t]")
    print(f"   Distribución: {df_targets['target_1d'].value_counts().to_dict()}")
    
    # Target 1 SEMANA: precio[t+5] > precio[t]
    df_targets['target_1w'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    print(f"\n📅 Target 1 SEMANA (t+5):")
    print(f"   Fórmula: precio[t+5] > precio[t]")
    print(f"   Distribución: {df_targets['target_1w'].value_counts().to_dict()}")
    
    # Target 1 MES: precio[t+20] > precio[t]
    df_targets['target_1m'] = (df['Close'].shift(-20) > df['Close']).astype(int)
    print(f"\n📅 Target 1 MES (t+20):")
    print(f"   Fórmula: precio[t+20] > precio[t]")
    print(f"   Distribución: {df_targets['target_1m'].value_counts().to_dict()}")
    
    # Eliminar filas con NaN en los targets
    rows_before = len(df_targets)
    df_targets = df_targets.dropna(subset=['target_1d', 'target_1w', 'target_1m'])
    rows_after = len(df_targets)
    
    print(f"\n📊 Resumen:")
    print(f"   Filas antes: {rows_before}")
    print(f"   Filas después: {rows_after}")
    print(f"   Filas eliminadas: {rows_before - rows_after}")
    
    return df_targets

def create_window_features(df):
    """
    Crea características basadas en ventanas temporales
    """
    WINDOW_1W = 5
    WINDOW_1M = 20
    
    df_windowed = df.copy()
    
    # VENTANA 1 SEMANA
    df_windowed['return_mean_1w'] = df['log_return'].rolling(WINDOW_1W).mean().shift(1)
    df_windowed['return_std_1w'] = df['log_return'].rolling(WINDOW_1W).std().shift(1)
    df_windowed['volume_mean_1w'] = df['Volume'].rolling(WINDOW_1W).mean().shift(1)
    df_windowed['volatility_mean_1w'] = df['volatility_7d'].rolling(WINDOW_1W).mean().shift(1)
    df_windowed['rsi_mean_1w'] = df['rsi_14'].rolling(WINDOW_1W).mean().shift(1)
    df_windowed['momentum_1w'] = (df['Close'] / df['Close'].shift(WINDOW_1W) - 1).shift(1)
    
    if 'mean_tone_shifted' in df.columns:
        df_windowed['sentiment_mean_1w'] = df['mean_tone_shifted'].rolling(WINDOW_1W).mean().shift(1)
    
    if 'n_news_shifted' in df.columns:
        df_windowed['news_sum_1w'] = df['n_news_shifted'].rolling(WINDOW_1W).sum().shift(1)
    
    # VENTANA 1 MES
    df_windowed['return_mean_1m'] = df['log_return'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['return_std_1m'] = df['log_return'].rolling(WINDOW_1M).std().shift(1)
    df_windowed['volume_mean_1m'] = df['Volume'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['volatility_mean_1m'] = df['volatility_7d'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['rsi_mean_1m'] = df['rsi_14'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['momentum_1m'] = (df['Close'] / df['Close'].shift(WINDOW_1M) - 1).shift(1)
    
    if 'mean_tone_shifted' in df.columns:
        df_windowed['sentiment_mean_1m'] = df['mean_tone_shifted'].rolling(WINDOW_1M).mean().shift(1)
    
    if 'n_news_shifted' in df.columns:
        df_windowed['news_sum_1m'] = df['n_news_shifted'].rolling(WINDOW_1M).sum().shift(1)
    
    df_windowed = df_windowed.dropna()
    
    return df_windowed

def get_features(df):
    """
    Obtiene las features para el modelo
    """
    exclude_cols = ['Date', 'date', 'Unnamed: 0', 'target', 'target_1d', 'target_1w', 'target_1m']
    features = [col for col in df.columns if col not in exclude_cols]
    return features

def train_and_evaluate_horizon(X_train, y_train, X_test, y_test, horizon_name):
    """
    Entrena y evalúa un modelo para un horizonte específico
    """
    print(f"\n{'='*80}")
    print(f"ENTRENANDO MODELO PARA HORIZONTE: {horizon_name}")
    print(f"{'='*80}")
    
    # Entrenar modelo
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Métricas
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    
    print(f"\n📊 MÉTRICAS:")
    print(f"   Accuracy Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Accuracy Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\n🎯 Métricas para clase SUBE (1):")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1-Score:  {test_f1:.4f}")
    
    # Reporte de clasificación
    print(f"\n📋 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_test_pred, target_names=['Baja (0)', 'Sube (1)']))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"📊 MATRIZ DE CONFUSIÓN:")
    print(f"                Predicho: Baja  Predicho: Sube")
    print(f"Real: Baja (0)      {cm[0][0]:6d}          {cm[0][1]:6d}")
    print(f"Real: Sube (1)      {cm[1][0]:6d}          {cm[1][1]:6d}")
    
    return {
        'horizon': horizon_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'confusion_matrix': cm
    }

def run_multi_horizon_study(df, features):
    """
    Ejecuta el estudio completo para múltiples horizontes
    """
    print("\n" + "="*80)
    print("ENTRENAMIENTO DE MODELOS MULTI-HORIZONTE")
    print("="*80)
    
    results = []
    
    # Preparar features
    X = df[features].values
    
    # Split común para todos los modelos
    test_size = 0.2
    random_state = 42
    
    # Scaler común
    scaler = StandardScaler()
    
    # ========================================
    # MODELO 1: Horizonte 1 DÍA
    # ========================================
    print(f"\n{'🔵'*40}")
    print("MODELO 1: PREDICCIÓN A 1 DÍA")
    print(f"{'🔵'*40}")
    
    y_1d = df['target_1d'].values
    X_train_1d, X_test_1d, y_train_1d, y_test_1d = train_test_split(
        X, y_1d, test_size=test_size, random_state=random_state, shuffle=False
    )
    X_train_1d_scaled = scaler.fit_transform(X_train_1d)
    X_test_1d_scaled = scaler.transform(X_test_1d)
    
    result_1d = train_and_evaluate_horizon(
        X_train_1d_scaled, y_train_1d, X_test_1d_scaled, y_test_1d, "1 DÍA (t+1)"
    )
    results.append(result_1d)
    
    # ========================================
    # MODELO 2: Horizonte 1 SEMANA
    # ========================================
    print(f"\n{'🟡'*40}")
    print("MODELO 2: PREDICCIÓN A 1 SEMANA")
    print(f"{'🟡'*40}")
    
    y_1w = df['target_1w'].values
    X_train_1w, X_test_1w, y_train_1w, y_test_1w = train_test_split(
        X, y_1w, test_size=test_size, random_state=random_state, shuffle=False
    )
    X_train_1w_scaled = scaler.fit_transform(X_train_1w)
    X_test_1w_scaled = scaler.transform(X_test_1w)
    
    result_1w = train_and_evaluate_horizon(
        X_train_1w_scaled, y_train_1w, X_test_1w_scaled, y_test_1w, "1 SEMANA (t+5)"
    )
    results.append(result_1w)
    
    # ========================================
    # MODELO 3: Horizonte 1 MES
    # ========================================
    print(f"\n{'🟢'*40}")
    print("MODELO 3: PREDICCIÓN A 1 MES")
    print(f"{'🟢'*40}")
    
    y_1m = df['target_1m'].values
    X_train_1m, X_test_1m, y_train_1m, y_test_1m = train_test_split(
        X, y_1m, test_size=test_size, random_state=random_state, shuffle=False
    )
    X_train_1m_scaled = scaler.fit_transform(X_train_1m)
    X_test_1m_scaled = scaler.transform(X_test_1m)
    
    result_1m = train_and_evaluate_horizon(
        X_train_1m_scaled, y_train_1m, X_test_1m_scaled, y_test_1m, "1 MES (t+20)"
    )
    results.append(result_1m)
    
    return pd.DataFrame(results)

def plot_multi_horizon_results(results_df, save_path='multi_horizon_results.png'):
    """
    Visualiza los resultados de múltiples horizontes
    """
    print("\n📊 Generando visualizaciones...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    horizons = results_df['horizon'].values
    colors = ['#3498db', '#f39c12', '#2ecc71']  # Azul, Naranja, Verde
    
    # Gráfico 1: Accuracy
    axes[0, 0].bar(horizons, results_df['test_accuracy'], color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Accuracy por Horizonte Temporal', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(results_df['test_accuracy']):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Gráfico 2: F1-Score
    axes[0, 1].bar(horizons, results_df['f1_score'], color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('F1-Score', fontsize=12)
    axes[0, 1].set_title('F1-Score por Horizonte Temporal', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(results_df['f1_score']):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Gráfico 3: Precision vs Recall
    x = np.arange(len(horizons))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, results_df['precision'], width, label='Precision', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 0].bar(x + width/2, results_df['recall'], width, label='Recall', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Precision vs Recall por Horizonte', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(horizons)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Gráfico 4: Comparación de todas las métricas
    metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(horizons))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        axes[1, 1].bar(x + i*width, results_df[metric], width, label=label, alpha=0.7, edgecolor='black')
    
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Comparación de Todas las Métricas', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x + width * 1.5)
    axes[1, 1].set_xticklabels(horizons)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado en: {save_path}")
    plt.close()

def print_summary(results_df):
    """
    Imprime un resumen interpretativo de los resultados
    """
    print("\n" + "="*80)
    print("RESUMEN E INTERPRETACIÓN DE RESULTADOS")
    print("="*80)
    
    # Mejor horizonte por métrica
    best_acc = results_df.loc[results_df['test_accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    best_recall = results_df.loc[results_df['recall'].idxmax()]
    
    print(f"\n🏆 MEJOR HORIZONTE POR MÉTRICA:")
    print(f"\n   Mejor Accuracy: {best_acc['horizon']}")
    print(f"      Accuracy: {best_acc['test_accuracy']:.4f}")
    print(f"      F1-Score: {best_acc['f1_score']:.4f}")
    
    print(f"\n   Mejor F1-Score: {best_f1['horizon']}")
    print(f"      Accuracy: {best_f1['test_accuracy']:.4f}")
    print(f"      F1-Score: {best_f1['f1_score']:.4f}")
    
    print(f"\n   Mejor Recall: {best_recall['horizon']}")
    print(f"      Recall: {best_recall['recall']:.4f}")
    print(f"      F1-Score: {best_recall['f1_score']:.4f}")
    
    # Análisis de tendencias
    print(f"\n" + "="*80)
    print("💡 CONCLUSIONES PARA EL TFG")
    print("="*80)
    
    # Comparar horizontes
    acc_1d = results_df[results_df['horizon'] == '1 DÍA (t+1)']['test_accuracy'].values[0]
    acc_1w = results_df[results_df['horizon'] == '1 SEMANA (t+5)']['test_accuracy'].values[0]
    acc_1m = results_df[results_df['horizon'] == '1 MES (t+20)']['test_accuracy'].values[0]
    
    f1_1d = results_df[results_df['horizon'] == '1 DÍA (t+1)']['f1_score'].values[0]
    f1_1w = results_df[results_df['horizon'] == '1 SEMANA (t+5)']['f1_score'].values[0]
    f1_1m = results_df[results_df['horizon'] == '1 MES (t+20)']['f1_score'].values[0]
    
    print(f"\n1. DIFICULTAD DE PREDICCIÓN POR HORIZONTE:")
    print(f"   • 1 DÍA:    Accuracy={acc_1d:.4f}, F1={f1_1d:.4f}")
    print(f"   • 1 SEMANA: Accuracy={acc_1w:.4f}, F1={f1_1w:.4f}")
    print(f"   • 1 MES:    Accuracy={acc_1m:.4f}, F1={f1_1m:.4f}")
    
    if acc_1d > acc_1w and acc_1d > acc_1m:
        print(f"\n   → El corto plazo (1 día) es MÁS PREDECIBLE")
        print(f"   → A mayor horizonte, mayor incertidumbre")
    elif acc_1m > acc_1d and acc_1m > acc_1w:
        print(f"\n   → El largo plazo (1 mes) es MÁS PREDECIBLE")
        print(f"   → Las tendencias de largo plazo son más estables")
    else:
        print(f"\n   → El plazo medio (1 semana) ofrece el mejor balance")
    
    print(f"\n2. IMPLICACIONES PRÁCTICAS:")
    if f1_1d > 0.55:
        print(f"   ✓ El modelo de 1 día es útil para trading de corto plazo")
    if f1_1w > 0.55:
        print(f"   ✓ El modelo de 1 semana es útil para swing trading")
    if f1_1m > 0.55:
        print(f"   ✓ El modelo de 1 mes es útil para inversión a medio plazo")
    
    print(f"\n3. RECOMENDACIÓN:")
    if best_f1['horizon'] == '1 DÍA (t+1)':
        print(f"   → Usar el modelo de 1 DÍA para decisiones de trading diario")
    elif best_f1['horizon'] == '1 SEMANA (t+5)':
        print(f"   → Usar el modelo de 1 SEMANA para estrategias semanales")
    else:
        print(f"   → Usar el modelo de 1 MES para estrategias de medio plazo")

def main():
    """
    Función principal
    """
    print("="*80)
    print("MULTI-HORIZON PREDICTION - RANDOM FOREST")
    print("Predicción de movimientos de precio en diferentes horizontes temporales")
    print("="*80)
    
    DATA_PATH = 'data/dataset_nvda_lstm.csv'
    OUTPUT_CSV = 'multi_horizon_results.csv'
    OUTPUT_PLOT = 'multi_horizon_results.png'
    
    try:
        # 1. Cargar datos
        print(f"\n📂 Cargando datos desde {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        print(f"   ✓ Dataset cargado: {df.shape}")
        
        # 2. Crear targets multi-horizonte
        df_targets = create_multi_horizon_targets(df)
        
        # 3. Crear features de ventanas
        print(f"\n🔧 Creando features de ventanas temporales...")
        df_windowed = create_window_features(df_targets)
        print(f"   ✓ Dataset con ventanas: {df_windowed.shape}")
        
        # 4. Obtener features
        features = get_features(df_windowed)
        print(f"\n📊 Total de features: {len(features)}")
        
        # 5. Ejecutar estudio multi-horizonte
        results_df = run_multi_horizon_study(df_windowed, features)
        
        # 6. Guardar resultados
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Resultados guardados en: {OUTPUT_CSV}")
        
        # 7. Visualizar
        plot_multi_horizon_results(results_df, save_path=OUTPUT_PLOT)
        
        # 8. Imprimir resumen
        print_summary(results_df)
        
        print("\n" + "="*80)
        print("✅ ANÁLISIS MULTI-HORIZONTE COMPLETADO EXITOSAMENTE")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()