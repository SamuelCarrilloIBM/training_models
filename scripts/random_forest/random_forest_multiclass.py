#!/usr/bin/env python3
"""
MULTI-CLASS CLASSIFICATION - Random Forest
Clasifica movimientos de precio en 3 categorías:
- SUBIDA: retorno > +0.5%
- NEUTRAL: retorno entre -0.5% y +0.5%
- BAJADA: retorno < -0.5%
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

def create_multiclass_target(df, threshold=0.005):
    """
    Crea target multi-clase basado en el cambio porcentual del precio
    
    Clases:
    - 0: BAJADA (retorno < -threshold)
    - 1: NEUTRAL (retorno entre -threshold y +threshold)
    - 2: SUBIDA (retorno > +threshold)
    """
    print("\n" + "="*80)
    print("CREANDO TARGET MULTI-CLASE")
    print("="*80)
    
    df_target = df.copy()
    
    # Calcular retorno del día siguiente
    df_target['next_return'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
    
    # Crear target multi-clase
    def classify_return(ret):
        if pd.isna(ret):
            return np.nan
        elif ret > threshold:
            return 2  # SUBIDA
        elif ret < -threshold:
            return 0  # BAJADA
        else:
            return 1  # NEUTRAL
    
    df_target['target_multiclass'] = df_target['next_return'].apply(classify_return)
    
    print(f"\n📊 Configuración del target:")
    print(f"   Threshold: ±{threshold*100:.2f}%")
    print(f"\n   Clase 0 (BAJADA):  retorno < -{threshold*100:.2f}%")
    print(f"   Clase 1 (NEUTRAL): retorno entre ±{threshold*100:.2f}%")
    print(f"   Clase 2 (SUBIDA):  retorno > +{threshold*100:.2f}%")
    
    # Distribución de clases
    class_dist = df_target['target_multiclass'].value_counts().sort_index()
    total = class_dist.sum()
    
    print(f"\n📈 Distribución de clases:")
    print(f"   Clase 0 (BAJADA):  {class_dist.get(0, 0):4d} ({class_dist.get(0, 0)/total*100:5.2f}%)")
    print(f"   Clase 1 (NEUTRAL): {class_dist.get(1, 0):4d} ({class_dist.get(1, 0)/total*100:5.2f}%)")
    print(f"   Clase 2 (SUBIDA):  {class_dist.get(2, 0):4d} ({class_dist.get(2, 0)/total*100:5.2f}%)")
    print(f"   Total:             {total:4d}")
    
    # Eliminar filas con NaN
    rows_before = len(df_target)
    df_target = df_target.dropna(subset=['target_multiclass'])
    rows_after = len(df_target)
    
    print(f"\n📊 Limpieza de datos:")
    print(f"   Filas antes: {rows_before}")
    print(f"   Filas después: {rows_after}")
    print(f"   Filas eliminadas: {rows_before - rows_after}")
    
    return df_target

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
    exclude_cols = ['Date', 'date', 'Unnamed: 0', 'target', 'target_multiclass', 'next_return']
    features = [col for col in df.columns if col not in exclude_cols]
    return features

def train_and_evaluate_multiclass(X_train, y_train, X_test, y_test):
    """
    Entrena y evalúa un modelo Random Forest multi-clase
    """
    print(f"\n{'='*80}")
    print("ENTRENANDO MODELO RANDOM FOREST MULTI-CLASE")
    print(f"{'='*80}")
    
    print("\n🔧 Hiperparámetros:")
    print("   • n_estimators: 100")
    print("   • max_depth: 5")
    print("   • min_samples_split: 20")
    print("   • min_samples_leaf: 10")
    print("   • max_features: 'sqrt'")
    print("   • class_weight: 'balanced'")
    
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
    
    # Métricas generales
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # F1-Score por clase y promedio
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\n{'='*80}")
    print("MÉTRICAS DEL MODELO")
    print(f"{'='*80}")
    print(f"\n📊 Accuracy:")
    print(f"   Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print(f"\n📊 F1-Score:")
    print(f"   Macro (promedio simple):    {f1_macro:.4f}")
    print(f"   Weighted (promedio pesado): {f1_weighted:.4f}")
    
    # Reporte de clasificación detallado
    print(f"\n{'='*80}")
    print("REPORTE DE CLASIFICACIÓN DETALLADO")
    print(f"{'='*80}")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['BAJADA (0)', 'NEUTRAL (1)', 'SUBIDA (2)'],
                                digits=4))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"{'='*80}")
    print("MATRIZ DE CONFUSIÓN")
    print(f"{'='*80}")
    print(f"\n                  Predicho:")
    print(f"                  BAJADA  NEUTRAL  SUBIDA")
    print(f"Real: BAJADA      {cm[0][0]:6d}   {cm[0][1]:6d}  {cm[0][2]:6d}")
    print(f"Real: NEUTRAL     {cm[1][0]:6d}   {cm[1][1]:6d}  {cm[1][2]:6d}")
    print(f"Real: SUBIDA      {cm[2][0]:6d}   {cm[2][1]:6d}  {cm[2][2]:6d}")
    
    # Análisis de errores
    print(f"\n{'='*80}")
    print("ANÁLISIS DE ERRORES")
    print(f"{'='*80}")
    
    total_errors = len(y_test) - np.sum(np.diag(cm))
    print(f"\nTotal de errores: {total_errors} de {len(y_test)} ({total_errors/len(y_test)*100:.2f}%)")
    
    # Errores por clase
    for i, class_name in enumerate(['BAJADA', 'NEUTRAL', 'SUBIDA']):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        class_errors = class_total - class_correct
        print(f"\n{class_name}:")
        print(f"   Total: {class_total}")
        print(f"   Correctos: {class_correct} ({class_correct/class_total*100:.2f}%)")
        print(f"   Errores: {class_errors} ({class_errors/class_total*100:.2f}%)")
        
        # Desglose de errores
        if class_errors > 0:
            print(f"   Confundido con:")
            for j, other_class in enumerate(['BAJADA', 'NEUTRAL', 'SUBIDA']):
                if i != j and cm[i][j] > 0:
                    print(f"      {other_class}: {cm[i][j]} ({cm[i][j]/class_total*100:.2f}%)")
    
    return {
        'model': rf_model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_test_pred
    }

def plot_results(results, save_path='multiclass_results.png'):
    """
    Visualiza los resultados del modelo multi-clase
    """
    print("\n📊 Generando visualizaciones...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    cm = results['confusion_matrix']
    
    # Gráfico 1: Matriz de confusión (valores absolutos)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['BAJADA', 'NEUTRAL', 'SUBIDA'],
                yticklabels=['BAJADA', 'NEUTRAL', 'SUBIDA'],
                cbar_kws={'label': 'Número de predicciones'})
    axes[0, 0].set_xlabel('Predicción', fontsize=12)
    axes[0, 0].set_ylabel('Real', fontsize=12)
    axes[0, 0].set_title('Matriz de Confusión (Valores Absolutos)', fontsize=14, fontweight='bold')
    
    # Gráfico 2: Matriz de confusión (porcentajes)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0, 1],
                xticklabels=['BAJADA', 'NEUTRAL', 'SUBIDA'],
                yticklabels=['BAJADA', 'NEUTRAL', 'SUBIDA'],
                cbar_kws={'label': 'Porcentaje (%)'})
    axes[0, 1].set_xlabel('Predicción', fontsize=12)
    axes[0, 1].set_ylabel('Real', fontsize=12)
    axes[0, 1].set_title('Matriz de Confusión (Porcentajes por Fila)', fontsize=14, fontweight='bold')
    
    # Gráfico 3: Métricas por clase
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    classes = ['BAJADA', 'NEUTRAL', 'SUBIDA']
    precision_per_class = precision_score(results['y_test'], results['y_pred'], average=None)
    recall_per_class = recall_score(results['y_test'], results['y_pred'], average=None)
    f1_per_class = f1_score(results['y_test'], results['y_pred'], average=None)
    
    x = np.arange(len(classes))
    width = 0.25
    
    axes[1, 0].bar(x - width, precision_per_class, width, label='Precision', alpha=0.8, color='#3498db')
    axes[1, 0].bar(x, recall_per_class, width, label='Recall', alpha=0.8, color='#e74c3c')
    axes[1, 0].bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8, color='#2ecc71')
    
    axes[1, 0].set_xlabel('Clase', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Métricas por Clase', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Añadir valores sobre las barras
    for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        axes[1, 0].text(i - width, p + 0.02, f'{p:.3f}', ha='center', fontsize=9)
        axes[1, 0].text(i, r + 0.02, f'{r:.3f}', ha='center', fontsize=9)
        axes[1, 0].text(i + width, f + 0.02, f'{f:.3f}', ha='center', fontsize=9)
    
    # Gráfico 4: Distribución de predicciones vs real
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    real_dist = pd.Series(y_test).value_counts().sort_index()
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, [real_dist.get(i, 0) for i in range(3)], width, 
                   label='Real', alpha=0.8, color='#3498db', edgecolor='black')
    axes[1, 1].bar(x + width/2, [pred_dist.get(i, 0) for i in range(3)], width, 
                   label='Predicho', alpha=0.8, color='#e74c3c', edgecolor='black')
    
    axes[1, 1].set_xlabel('Clase', fontsize=12)
    axes[1, 1].set_ylabel('Frecuencia', fontsize=12)
    axes[1, 1].set_title('Distribución: Real vs Predicho', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(classes)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado en: {save_path}")
    plt.close()

def print_summary(results):
    """
    Imprime un resumen interpretativo de los resultados
    """
    print("\n" + "="*80)
    print("RESUMEN E INTERPRETACIÓN DE RESULTADOS")
    print("="*80)
    
    print(f"\n📊 RENDIMIENTO GENERAL:")
    print(f"   Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"   F1-Score (macro): {results['f1_macro']:.4f}")
    print(f"   F1-Score (weighted): {results['f1_weighted']:.4f}")
    
    # Análisis por clase
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision_per_class = precision_score(results['y_test'], results['y_pred'], average=None)
    recall_per_class = recall_score(results['y_test'], results['y_pred'], average=None)
    f1_per_class = f1_score(results['y_test'], results['y_pred'], average=None)
    
    print(f"\n📈 RENDIMIENTO POR CLASE:")
    
    classes = ['BAJADA', 'NEUTRAL', 'SUBIDA']
    for i, class_name in enumerate(classes):
        print(f"\n   {class_name}:")
        print(f"      Precision: {precision_per_class[i]:.4f}")
        print(f"      Recall:    {recall_per_class[i]:.4f}")
        print(f"      F1-Score:  {f1_per_class[i]:.4f}")
        
        if f1_per_class[i] > 0.6:
            print(f"      ✓ Buen rendimiento")
        elif f1_per_class[i] > 0.4:
            print(f"      ⚠️  Rendimiento moderado")
        else:
            print(f"      ❌ Rendimiento bajo")
    
    # Conclusiones
    print(f"\n" + "="*80)
    print("💡 CONCLUSIONES PARA EL TFG")
    print("="*80)
    
    best_class_idx = np.argmax(f1_per_class)
    worst_class_idx = np.argmin(f1_per_class)
    
    print(f"\n1. CLASE MÁS FÁCIL DE PREDECIR: {classes[best_class_idx]}")
    print(f"   F1-Score: {f1_per_class[best_class_idx]:.4f}")
    
    print(f"\n2. CLASE MÁS DIFÍCIL DE PREDECIR: {classes[worst_class_idx]}")
    print(f"   F1-Score: {f1_per_class[worst_class_idx]:.4f}")
    
    print(f"\n3. COMPARACIÓN CON CLASIFICACIÓN BINARIA:")
    print(f"   • La clasificación multi-clase es más realista")
    print(f"   • Permite identificar movimientos neutrales (sin tendencia clara)")
    print(f"   • Útil para estrategias que evitan operar en mercados laterales")
    
    if results['test_accuracy'] > 0.5:
        print(f"\n4. UTILIDAD PRÁCTICA:")
        print(f"   ✓ El modelo supera el azar (33.33% para 3 clases)")
        print(f"   ✓ Accuracy de {results['test_accuracy']*100:.2f}% indica capacidad predictiva")
    else:
        print(f"\n4. LIMITACIONES:")
        print(f"   ⚠️  Accuracy cercana al azar")
        print(f"   ⚠️  Se necesita más ingeniería de features o datos")

def main():
    """
    Función principal
    """
    print("="*80)
    print("MULTI-CLASS CLASSIFICATION - RANDOM FOREST")
    print("Clasificación de movimientos de precio en 3 categorías")
    print("="*80)
    
    DATA_PATH = 'data/dataset_nvda_lstm.csv'
    OUTPUT_CSV = 'multiclass_results.csv'
    OUTPUT_PLOT = 'multiclass_results.png'
    THRESHOLD = 0.005  # 0.5%
    
    try:
        # 1. Cargar datos
        print(f"\n📂 Cargando datos desde {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        print(f"   ✓ Dataset cargado: {df.shape}")
        
        # 2. Crear target multi-clase
        df_target = create_multiclass_target(df, threshold=THRESHOLD)
        
        # 3. Crear features de ventanas
        print(f"\n🔧 Creando features de ventanas temporales...")
        df_windowed = create_window_features(df_target)
        print(f"   ✓ Dataset con ventanas: {df_windowed.shape}")
        
        # 4. Obtener features
        features = get_features(df_windowed)
        print(f"\n📊 Total de features: {len(features)}")
        
        # 5. Preparar datos
        X = df_windowed[features].values
        y = df_windowed['target_multiclass'].values.astype(int)
        
        # 6. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"\n📊 División de datos:")
        print(f"   Train: {X_train.shape}")
        print(f"   Test:  {X_test.shape}")
        
        # 7. Escalar
        print(f"\n🔧 Escalando features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 8. Entrenar y evaluar
        results = train_and_evaluate_multiclass(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # 9. Guardar resultados
        results_df = pd.DataFrame([{
            'threshold': THRESHOLD,
            'train_accuracy': results['train_accuracy'],
            'test_accuracy': results['test_accuracy'],
            'f1_macro': results['f1_macro'],
            'f1_weighted': results['f1_weighted']
        }])
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Resultados guardados en: {OUTPUT_CSV}")
        
        # 10. Visualizar
        plot_results(results, save_path=OUTPUT_PLOT)
        
        # 11. Imprimir resumen
        print_summary(results)
        
        print("\n" + "="*80)
        print("✅ ANÁLISIS MULTI-CLASE COMPLETADO EXITOSAMENTE")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()