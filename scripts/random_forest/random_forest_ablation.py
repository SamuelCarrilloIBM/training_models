#!/usr/bin/env python3
"""
ABLATION STUDY - Random Forest
Compara el rendimiento de diferentes conjuntos de features para determinar
cuáles son más importantes para la predicción.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

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
    df_windowed['return_max_1w'] = df['log_return'].rolling(WINDOW_1W).max().shift(1)
    df_windowed['return_min_1w'] = df['log_return'].rolling(WINDOW_1W).min().shift(1)
    df_windowed['volume_mean_1w'] = df['Volume'].rolling(WINDOW_1W).mean().shift(1)
    df_windowed['volume_std_1w'] = df['Volume'].rolling(WINDOW_1W).std().shift(1)
    df_windowed['volatility_mean_1w'] = df['volatility_7d'].rolling(WINDOW_1W).mean().shift(1)
    df_windowed['rsi_mean_1w'] = df['rsi_14'].rolling(WINDOW_1W).mean().shift(1)
    df_windowed['momentum_1w'] = (df['Close'] / df['Close'].shift(WINDOW_1W) - 1).shift(1)
    
    if 'mean_tone_shifted' in df.columns:
        df_windowed['sentiment_mean_1w'] = df['mean_tone_shifted'].rolling(WINDOW_1W).mean().shift(1)
        df_windowed['sentiment_std_1w'] = df['mean_tone_shifted'].rolling(WINDOW_1W).std().shift(1)
    
    if 'n_news_shifted' in df.columns:
        df_windowed['news_sum_1w'] = df['n_news_shifted'].rolling(WINDOW_1W).sum().shift(1)
    
    # VENTANA 1 MES
    df_windowed['return_mean_1m'] = df['log_return'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['return_std_1m'] = df['log_return'].rolling(WINDOW_1M).std().shift(1)
    df_windowed['return_max_1m'] = df['log_return'].rolling(WINDOW_1M).max().shift(1)
    df_windowed['return_min_1m'] = df['log_return'].rolling(WINDOW_1M).min().shift(1)
    df_windowed['volume_mean_1m'] = df['Volume'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['volume_std_1m'] = df['Volume'].rolling(WINDOW_1M).std().shift(1)
    df_windowed['volatility_mean_1m'] = df['volatility_7d'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['rsi_mean_1m'] = df['rsi_14'].rolling(WINDOW_1M).mean().shift(1)
    df_windowed['momentum_1m'] = (df['Close'] / df['Close'].shift(WINDOW_1M) - 1).shift(1)
    
    if 'mean_tone_shifted' in df.columns:
        df_windowed['sentiment_mean_1m'] = df['mean_tone_shifted'].rolling(WINDOW_1M).mean().shift(1)
        df_windowed['sentiment_std_1m'] = df['mean_tone_shifted'].rolling(WINDOW_1M).std().shift(1)
    
    if 'n_news_shifted' in df.columns:
        df_windowed['news_sum_1m'] = df['n_news_shifted'].rolling(WINDOW_1M).sum().shift(1)
    
    # RATIOS
    df_windowed['volatility_ratio_1w_1m'] = (
        df_windowed['return_std_1w'] / df_windowed['return_std_1m']
    )
    df_windowed['volume_ratio_1w_1m'] = (
        df_windowed['volume_mean_1w'] / df_windowed['volume_mean_1m']
    )
    df_windowed['momentum_diff_1w_1m'] = (
        df_windowed['momentum_1w'] - df_windowed['momentum_1m']
    )
    
    df_windowed = df_windowed.dropna()
    
    return df_windowed

def get_feature_groups(df):
    """
    Organiza las features en grupos
    """
    all_features = [col for col in df.columns if col not in ['Date', 'date', 'Unnamed: 0', 'target']]
    
    base_features = [col for col in all_features if not any(x in col for x in ['_1w', '_1m', 'lag_'])]
    lag_features = [col for col in all_features if 'lag_' in col]
    window_1w_features = [col for col in all_features if '_1w' in col]
    window_1m_features = [col for col in all_features if '_1m' in col]
    
    return {
        'base': base_features,
        'lags': lag_features,
        'window_1w': window_1w_features,
        'window_1m': window_1m_features,
        'all': all_features
    }

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    """
    Entrena y evalúa un modelo Random Forest
    """
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
    y_pred = rf_model.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_features': X_train.shape[1]
    }

def run_ablation_study(df, feature_groups):
    """
    Ejecuta el estudio de ablación completo
    """
    print("\n" + "="*80)
    print("ABLATION STUDY - ANÁLISIS DE IMPORTANCIA DE GRUPOS DE FEATURES")
    print("="*80)
    
    # Preparar datos
    X_all = df[feature_groups['all']].values
    y = df['target'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    # ========================================
    # PARTE 1: MODELOS INDIVIDUALES
    # ========================================
    print("\n" + "="*80)
    print("PARTE 1: MODELOS CON GRUPOS INDIVIDUALES DE FEATURES")
    print("="*80)
    
    # Modelo A: Solo BASE
    print("\n🔵 Modelo A: Solo features BASE...")
    base_indices = [i for i, col in enumerate(feature_groups['all']) if col in feature_groups['base']]
    X_train_base = X_train_scaled[:, base_indices]
    X_test_base = X_test_scaled[:, base_indices]
    result_base = train_and_evaluate(X_train_base, y_train, X_test_base, y_test, "A: Solo BASE")
    results.append(result_base)
    print(f"   ✓ Accuracy: {result_base['accuracy']:.4f} | F1: {result_base['f1_score']:.4f} | Recall: {result_base['recall']:.4f}")
    
    # Modelo B: Solo LAGS
    print("\n🟡 Modelo B: Solo features LAGS...")
    lag_indices = [i for i, col in enumerate(feature_groups['all']) if col in feature_groups['lags']]
    X_train_lags = X_train_scaled[:, lag_indices]
    X_test_lags = X_test_scaled[:, lag_indices]
    result_lags = train_and_evaluate(X_train_lags, y_train, X_test_lags, y_test, "B: Solo LAGS")
    results.append(result_lags)
    print(f"   ✓ Accuracy: {result_lags['accuracy']:.4f} | F1: {result_lags['f1_score']:.4f} | Recall: {result_lags['recall']:.4f}")
    
    # Modelo C: Solo 1 SEMANA
    print("\n🔵 Modelo C: Solo features 1 SEMANA...")
    w1w_indices = [i for i, col in enumerate(feature_groups['all']) if col in feature_groups['window_1w']]
    X_train_1w = X_train_scaled[:, w1w_indices]
    X_test_1w = X_test_scaled[:, w1w_indices]
    result_1w = train_and_evaluate(X_train_1w, y_train, X_test_1w, y_test, "C: Solo 1 SEMANA")
    results.append(result_1w)
    print(f"   ✓ Accuracy: {result_1w['accuracy']:.4f} | F1: {result_1w['f1_score']:.4f} | Recall: {result_1w['recall']:.4f}")
    
    # Modelo D: Solo 1 MES
    print("\n🟢 Modelo D: Solo features 1 MES...")
    w1m_indices = [i for i, col in enumerate(feature_groups['all']) if col in feature_groups['window_1m']]
    X_train_1m = X_train_scaled[:, w1m_indices]
    X_test_1m = X_test_scaled[:, w1m_indices]
    result_1m = train_and_evaluate(X_train_1m, y_train, X_test_1m, y_test, "D: Solo 1 MES")
    results.append(result_1m)
    print(f"   ✓ Accuracy: {result_1m['accuracy']:.4f} | F1: {result_1m['f1_score']:.4f} | Recall: {result_1m['recall']:.4f}")
    
    # ========================================
    # PARTE 2: COMBINACIONES
    # ========================================
    print("\n" + "="*80)
    print("PARTE 2: MODELOS CON COMBINACIONES DE FEATURES")
    print("="*80)
    
    # Modelo E: BASE + LAGS
    print("\n⚪ Modelo E: BASE + LAGS...")
    base_lag_indices = [i for i, col in enumerate(feature_groups['all']) 
                        if col in feature_groups['base'] or col in feature_groups['lags']]
    X_train_base_lag = X_train_scaled[:, base_lag_indices]
    X_test_base_lag = X_test_scaled[:, base_lag_indices]
    result_base_lag = train_and_evaluate(X_train_base_lag, y_train, X_test_base_lag, y_test, "E: BASE + LAGS")
    results.append(result_base_lag)
    print(f"   ✓ Accuracy: {result_base_lag['accuracy']:.4f} | F1: {result_base_lag['f1_score']:.4f} | Recall: {result_base_lag['recall']:.4f}")
    
    # Modelo F: TODAS
    print("\n⭐ Modelo F: TODAS las features...")
    result_all = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, "F: TODAS")
    results.append(result_all)
    print(f"   ✓ Accuracy: {result_all['accuracy']:.4f} | F1: {result_all['f1_score']:.4f} | Recall: {result_all['recall']:.4f}")
    
    # ========================================
    # PARTE 3: ABLACIÓN (quitar grupos)
    # ========================================
    print("\n" + "="*80)
    print("PARTE 3: ABLACIÓN - QUITANDO GRUPOS DE FEATURES")
    print("="*80)
    
    # Modelo G: SIN LAGS
    print("\n❌ Modelo G: SIN LAGS (todas menos lags)...")
    no_lags_indices = [i for i, col in enumerate(feature_groups['all']) if col not in feature_groups['lags']]
    X_train_no_lags = X_train_scaled[:, no_lags_indices]
    X_test_no_lags = X_test_scaled[:, no_lags_indices]
    result_no_lags = train_and_evaluate(X_train_no_lags, y_train, X_test_no_lags, y_test, "G: SIN LAGS")
    results.append(result_no_lags)
    print(f"   ✓ Accuracy: {result_no_lags['accuracy']:.4f} | F1: {result_no_lags['f1_score']:.4f} | Recall: {result_no_lags['recall']:.4f}")
    print(f"   📉 Caída vs TODAS: Acc={result_all['accuracy']-result_no_lags['accuracy']:.4f}, F1={result_all['f1_score']-result_no_lags['f1_score']:.4f}")
    
    # Modelo H: SIN 1 SEMANA
    print("\n❌ Modelo H: SIN 1 SEMANA (todas menos 1w)...")
    no_1w_indices = [i for i, col in enumerate(feature_groups['all']) if col not in feature_groups['window_1w']]
    X_train_no_1w = X_train_scaled[:, no_1w_indices]
    X_test_no_1w = X_test_scaled[:, no_1w_indices]
    result_no_1w = train_and_evaluate(X_train_no_1w, y_train, X_test_no_1w, y_test, "H: SIN 1 SEMANA")
    results.append(result_no_1w)
    print(f"   ✓ Accuracy: {result_no_1w['accuracy']:.4f} | F1: {result_no_1w['f1_score']:.4f} | Recall: {result_no_1w['recall']:.4f}")
    print(f"   📉 Caída vs TODAS: Acc={result_all['accuracy']-result_no_1w['accuracy']:.4f}, F1={result_all['f1_score']-result_no_1w['f1_score']:.4f}")
    
    # Modelo I: SIN 1 MES
    print("\n❌ Modelo I: SIN 1 MES (todas menos 1m)...")
    no_1m_indices = [i for i, col in enumerate(feature_groups['all']) if col not in feature_groups['window_1m']]
    X_train_no_1m = X_train_scaled[:, no_1m_indices]
    X_test_no_1m = X_test_scaled[:, no_1m_indices]
    result_no_1m = train_and_evaluate(X_train_no_1m, y_train, X_test_no_1m, y_test, "I: SIN 1 MES")
    results.append(result_no_1m)
    print(f"   ✓ Accuracy: {result_no_1m['accuracy']:.4f} | F1: {result_no_1m['f1_score']:.4f} | Recall: {result_no_1m['recall']:.4f}")
    print(f"   📉 Caída vs TODAS: Acc={result_all['accuracy']-result_no_1m['accuracy']:.4f}, F1={result_all['f1_score']-result_no_1m['f1_score']:.4f}")
    
    return pd.DataFrame(results)

def plot_ablation_results(results_df, save_path='ablation_study_results.png'):
    """
    Visualiza los resultados del estudio de ablación
    """
    print("\n📊 Generando visualizaciones...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colores por categoría
    colors = []
    for name in results_df['model_name']:
        if 'Solo' in name:
            colors.append('#3498db')  # Azul para individuales
        elif 'SIN' in name:
            colors.append('#e74c3c')  # Rojo para ablación
        elif 'TODAS' in name:
            colors.append('#2ecc71')  # Verde para completo
        else:
            colors.append('#f39c12')  # Naranja para combinaciones
    
    # Gráfico 1: Accuracy
    axes[0, 0].barh(range(len(results_df)), results_df['accuracy'], color=colors)
    axes[0, 0].set_yticks(range(len(results_df)))
    axes[0, 0].set_yticklabels(results_df['model_name'], fontsize=9)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Accuracy por Modelo', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    axes[0, 0].axvline(x=results_df[results_df['model_name'].str.contains('TODAS')]['accuracy'].values[0], 
                       color='green', linestyle='--', alpha=0.5, label='Baseline (TODAS)')
    axes[0, 0].legend()
    
    # Gráfico 2: F1-Score
    axes[0, 1].barh(range(len(results_df)), results_df['f1_score'], color=colors)
    axes[0, 1].set_yticks(range(len(results_df)))
    axes[0, 1].set_yticklabels(results_df['model_name'], fontsize=9)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel('F1-Score', fontsize=11)
    axes[0, 1].set_title('F1-Score por Modelo', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    axes[0, 1].axvline(x=results_df[results_df['model_name'].str.contains('TODAS')]['f1_score'].values[0], 
                       color='green', linestyle='--', alpha=0.5, label='Baseline (TODAS)')
    axes[0, 1].legend()
    
    # Gráfico 3: Recall
    axes[1, 0].barh(range(len(results_df)), results_df['recall'], color=colors)
    axes[1, 0].set_yticks(range(len(results_df)))
    axes[1, 0].set_yticklabels(results_df['model_name'], fontsize=9)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel('Recall', fontsize=11)
    axes[1, 0].set_title('Recall por Modelo', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].axvline(x=results_df[results_df['model_name'].str.contains('TODAS')]['recall'].values[0], 
                       color='green', linestyle='--', alpha=0.5, label='Baseline (TODAS)')
    axes[1, 0].legend()
    
    # Gráfico 4: Comparación de métricas
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[1, 1].bar(x + i*width, results_df[metric], width, label=metric.capitalize())
    
    axes[1, 1].set_xlabel('Modelos', fontsize=11)
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('Comparación de Todas las Métricas', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x + width * 1.5)
    axes[1, 1].set_xticklabels([name.split(':')[0] for name in results_df['model_name']], rotation=45, ha='right')
    axes[1, 1].legend()
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
    
    # Mejor modelo individual
    individual_models = results_df[results_df['model_name'].str.contains('Solo')]
    best_individual = individual_models.loc[individual_models['f1_score'].idxmax()]
    
    print(f"\n🏆 MEJOR MODELO INDIVIDUAL:")
    print(f"   {best_individual['model_name']}")
    print(f"   Accuracy: {best_individual['accuracy']:.4f}")
    print(f"   F1-Score: {best_individual['f1_score']:.4f}")
    print(f"   Recall: {best_individual['recall']:.4f}")
    
    # Modelo completo
    full_model = results_df[results_df['model_name'].str.contains('TODAS')].iloc[0]
    
    print(f"\n⭐ MODELO COMPLETO (TODAS):")
    print(f"   Accuracy: {full_model['accuracy']:.4f}")
    print(f"   F1-Score: {full_model['f1_score']:.4f}")
    print(f"   Recall: {full_model['recall']:.4f}")
    
    # Análisis de ablación
    ablation_models = results_df[results_df['model_name'].str.contains('SIN')]
    
    print(f"\n📉 IMPACTO DE QUITAR GRUPOS (vs TODAS):")
    for _, row in ablation_models.iterrows():
        acc_drop = full_model['accuracy'] - row['accuracy']
        f1_drop = full_model['f1_score'] - row['f1_score']
        
        print(f"\n   {row['model_name']}:")
        print(f"      Caída Accuracy: {acc_drop:+.4f} ({acc_drop/full_model['accuracy']*100:+.2f}%)")
        print(f"      Caída F1-Score: {f1_drop:+.4f} ({f1_drop/full_model['f1_score']*100:+.2f}%)")
        
        if abs(f1_drop) > 0.05:
            print(f"      ⚠️  IMPACTO ALTO - Este grupo es MUY importante")
        elif abs(f1_drop) > 0.02:
            print(f"      ⚡ IMPACTO MODERADO - Este grupo aporta valor")
        else:
            print(f"      ✓ IMPACTO BAJO - Este grupo es menos crítico")
    
    # Conclusión
    print(f"\n" + "="*80)
    print("💡 CONCLUSIONES PARA EL TFG")
    print("="*80)
    
    # Identificar el grupo más importante
    max_drop = 0
    most_important = ""
    for _, row in ablation_models.iterrows():
        f1_drop = full_model['f1_score'] - row['f1_score']
        if f1_drop > max_drop:
            max_drop = f1_drop
            most_important = row['model_name'].replace('SIN ', '')
    
    print(f"\n1. El grupo de features MÁS IMPORTANTE es: {most_important}")
    print(f"   (Quitarlo causa una caída de {max_drop:.4f} en F1-Score)")
    
    print(f"\n2. Comparación de ventanas temporales:")
    solo_1w = results_df[results_df['model_name'].str.contains('Solo 1 SEMANA')].iloc[0]
    solo_1m = results_df[results_df['model_name'].str.contains('Solo 1 MES')].iloc[0]
    
    if solo_1w['f1_score'] > solo_1m['f1_score']:
        print(f"   → La ventana de 1 SEMANA predice mejor que 1 MES")
        print(f"   → Esto sugiere que la información de corto plazo es más relevante")
    else:
        print(f"   → La ventana de 1 MES predice mejor que 1 SEMANA")
        print(f"   → Esto sugiere que las tendencias de largo plazo son más importantes")
    
    print(f"\n3. Valor de combinar features:")
    improvement = full_model['f1_score'] - best_individual['f1_score']
    print(f"   → Combinar todos los grupos mejora el F1-Score en {improvement:.4f}")
    print(f"   → Esto demuestra que diferentes escalas temporales aportan información complementaria")

def main():
    """
    Función principal
    """
    print("="*80)
    print("ABLATION STUDY - RANDOM FOREST")
    print("Análisis del impacto de diferentes grupos de features")
    print("="*80)
    
    DATA_PATH = 'data/dataset_nvda_lstm.csv'
    OUTPUT_CSV = 'ablation_study_results.csv'
    OUTPUT_PLOT = 'ablation_study_results.png'
    
    try:
        # 1. Cargar datos
        print(f"\n📂 Cargando datos desde {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        print(f"   ✓ Dataset cargado: {df.shape}")
        
        # 2. Crear features de ventanas
        print(f"\n🔧 Creando features de ventanas temporales...")
        df_windowed = create_window_features(df)
        print(f"   ✓ Dataset con ventanas: {df_windowed.shape}")
        
        # 3. Obtener grupos de features
        feature_groups = get_feature_groups(df_windowed)
        print(f"\n📊 Grupos de features identificados:")
        print(f"   • BASE: {len(feature_groups['base'])} features")
        print(f"   • LAGS: {len(feature_groups['lags'])} features")
        print(f"   • 1 SEMANA: {len(feature_groups['window_1w'])} features")
        print(f"   • 1 MES: {len(feature_groups['window_1m'])} features")
        print(f"   • TOTAL: {len(feature_groups['all'])} features")
        
        # 4. Ejecutar ablation study
        results_df = run_ablation_study(df_windowed, feature_groups)
        
        # 5. Guardar resultados
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Resultados guardados en: {OUTPUT_CSV}")
        
        # 6. Visualizar
        plot_ablation_results(results_df, save_path=OUTPUT_PLOT)
        
        # 7. Imprimir resumen
        print_summary(results_df)
        
        print("\n" + "="*80)
        print("✅ ABLATION STUDY COMPLETADO EXITOSAMENTE")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()