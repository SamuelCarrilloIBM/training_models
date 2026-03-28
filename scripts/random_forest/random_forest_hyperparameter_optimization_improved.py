#!/usr/bin/env python3
"""
Random Forest Hyperparameter Optimization - VERSIÓN MEJORADA
=============================================================

Mejoras implementadas:
1. ✅ Eliminado StandardScaler (RF no lo necesita)
2. ✅ Métrica optimizada: fbeta_score (beta=2 para priorizar recall)
3. ✅ TimeSeriesSplit aumentado a 5 splits
4. ✅ Feature engineering avanzado (lags, rolling, momentum)
5. ✅ Threshold optimization con 100 puntos (más fino)
6. ✅ Espacio de búsqueda de Optuna mejorado
7. ✅ Número de trials aumentado a 200
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    fbeta_score,
    make_scorer
)
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATA_PATH = 'data/dataset_nvda_lstm.csv'
RANDOM_STATE = 42
N_SPLITS = 5  # ✅ Aumentado de 3 a 5
N_TRIALS_OPTUNA = 200  # ✅ Aumentado de 50 a 200

# ============================================================================
# FUNCIONES DE FEATURE ENGINEERING
# ============================================================================

def add_advanced_features(df):
    """
    Añade features avanzadas: lags, rolling stats, momentum
    """
    df = df.copy()
    
    # Detectar columna de precio (puede ser 'close', 'Close', o similar)
    price_col = None
    for col in ['close', 'Close', 'precio', 'price']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        print("⚠️ No se encontró columna de precio. Usando features existentes sin feature engineering adicional.")
        return df
    
    print(f"📊 Usando columna de precio: '{price_col}'")
    
    # 1. LAGS (retornos pasados)
    print("📊 Añadiendo lags...")
    df['return_1'] = df[price_col].pct_change()
    df['return_5'] = df[price_col].pct_change(5)
    df['return_10'] = df[price_col].pct_change(10)
    df['return_20'] = df[price_col].pct_change(20)
    
    # 2. ROLLING STATISTICS
    print("📊 Añadiendo rolling statistics...")
    for window in [5, 10, 20]:
        df[f'ma_{window}'] = df[price_col].rolling(window).mean()
        df[f'std_{window}'] = df[price_col].rolling(window).std()
        df[f'min_{window}'] = df[price_col].rolling(window).min()
        df[f'max_{window}'] = df[price_col].rolling(window).max()
    
    # 3. MOMENTUM
    print("📊 Añadiendo momentum...")
    df['momentum_5'] = df[price_col] - df[price_col].shift(5)
    df['momentum_10'] = df[price_col] - df[price_col].shift(10)
    df['momentum_20'] = df[price_col] - df[price_col].shift(20)
    
    # 4. VOLATILIDAD
    print("📊 Añadiendo volatilidad...")
    df['volatility_5'] = df['return_1'].rolling(5).std()
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()
    
    # 5. RATIOS
    print("📊 Añadiendo ratios...")
    df['price_to_ma_10'] = df[price_col] / df['ma_10']
    df['price_to_ma_20'] = df[price_col] / df['ma_20']
    
    # 6. FEATURES DE SENTIMIENTO (si existen)
    sentiment_cols = [col for col in df.columns if 'tone' in col.lower() or 'sentiment' in col.lower()]
    if sentiment_cols:
        print(f"📊 Añadiendo features de sentimiento basadas en: {sentiment_cols[:3]}")
        for col in sentiment_cols[:3]:  # Usar las primeras 3 columnas de sentimiento
            df[f'{col}_ma_5'] = df[col].rolling(5).mean()
            df[f'{col}_std_5'] = df[col].rolling(5).std()
    
    # Eliminar NaN generados por rolling/lags
    df = df.dropna()
    
    print(f"✅ Features añadidas. Shape final: {df.shape}")
    return df

# ============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

def load_and_prepare_data():
    """Carga y prepara los datos con feature engineering avanzado"""
    print("=" * 80)
    print("📂 CARGANDO DATOS")
    print("=" * 80)
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Datos cargados: {df.shape}")
    print(f"📅 Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    
    # Aplicar feature engineering avanzado
    print("\n" + "=" * 80)
    print("🔧 FEATURE ENGINEERING AVANZADO")
    print("=" * 80)
    df = add_advanced_features(df)
    
    # Separar features y target
    target_col = 'target'
    
    # Detectar columnas a excluir (fecha, target, precio original)
    exclude_cols = [target_col]
    for col in ['date', 'Date', 'fecha', 'close', 'Close', 'precio', 'price']:
        if col in df.columns:
            exclude_cols.append(col)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"\n✅ Features finales: {len(feature_cols)}")
    print(f"✅ Muestras: {len(X)}")
    print(f"✅ Distribución del target: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

# ============================================================================
# OPTIMIZACIÓN DE THRESHOLD
# ============================================================================

def optimize_threshold(model, X, y, cv):
    """
    Optimiza el threshold usando validación cruzada
    ✅ Mejora: 100 puntos en lugar de grid grueso
    """
    print("\n" + "=" * 80)
    print("🎯 OPTIMIZACIÓN DE THRESHOLD (100 puntos)")
    print("=" * 80)
    
    thresholds = np.linspace(0.1, 0.9, 100)  # ✅ Mejora: más fino
    best_threshold = 0.5
    best_f1 = 0
    
    results = []
    
    for threshold in thresholds:
        f1_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
        
        mean_f1 = np.mean(f1_scores)
        results.append({
            'threshold': threshold,
            'f1_score': mean_f1
        })
        
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_threshold = threshold
    
    print(f"✅ Mejor threshold: {best_threshold:.3f}")
    print(f"✅ F1-Score: {best_f1:.4f}")
    
    # Visualización
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['threshold'], results_df['f1_score'], linewidth=2)
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Óptimo: {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('Optimización de Threshold (100 puntos)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('threshold_optimization_improved.png', dpi=300)
    print("💾 Gráfico guardado: threshold_optimization_improved.png")
    
    return best_threshold, results_df

# ============================================================================
# GRID SEARCH
# ============================================================================

def run_grid_search(X, y, cv):
    """Grid Search con fbeta_score"""
    print("\n" + "=" * 80)
    print("🔍 GRID SEARCH")
    print("=" * 80)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # ✅ Mejora: fbeta_score con beta=2 (prioriza recall)
    fbeta_scorer = make_scorer(fbeta_score, beta=2)
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=cv, 
        scoring=fbeta_scorer,  # ✅ Mejora
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"\n✅ Mejores parámetros: {grid_search.best_params_}")
    print(f"✅ Mejor F-beta score (CV): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

# ============================================================================
# RANDOM SEARCH
# ============================================================================

def run_random_search(X, y, cv):
    """Random Search con fbeta_score"""
    print("\n" + "=" * 80)
    print("🎲 RANDOM SEARCH")
    print("=" * 80)
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # ✅ Mejora: fbeta_score con beta=2
    fbeta_scorer = make_scorer(fbeta_score, beta=2)
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        rf, 
        param_distributions, 
        n_iter=100,
        cv=cv, 
        scoring=fbeta_scorer,  # ✅ Mejora
        n_jobs=-1, 
        verbose=1,
        random_state=RANDOM_STATE
    )
    
    random_search.fit(X, y)
    
    print(f"\n✅ Mejores parámetros: {random_search.best_params_}")
    print(f"✅ Mejor F-beta score (CV): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_

# ============================================================================
# OPTUNA
# ============================================================================

def run_optuna_optimization(X, y, cv):
    """
    Optuna con espacio de búsqueda mejorado y más trials
    ✅ Mejoras: mejor search space y 200 trials
    """
    print("\n" + "=" * 80)
    print(f"🔬 OPTUNA OPTIMIZATION ({N_TRIALS_OPTUNA} trials)")
    print("=" * 80)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),  # ✅ Mejora: rango más acotado
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),  # ✅ Mejora
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        
        rf = RandomForestClassifier(**params)
        
        # ✅ Mejora: fbeta_score con beta=2
        fbeta_scorer = make_scorer(fbeta_score, beta=2)
        scores = cross_val_score(rf, X, y, cv=cv, scoring=fbeta_scorer, n_jobs=-1)
        
        return scores.mean()
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)
    
    print(f"\n✅ Mejores parámetros: {study.best_params}")
    print(f"✅ Mejor F-beta score (CV): {study.best_value:.4f}")
    
    # Entrenar modelo final con mejores parámetros
    best_params = study.best_params
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    
    best_model = RandomForestClassifier(**best_params)
    
    return best_model, study.best_params

# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate_model(model, X, y, cv, method_name, threshold=0.5):
    """Evalúa el modelo con validación cruzada"""
    print(f"\n{'=' * 80}")
    print(f"📊 EVALUACIÓN: {method_name}")
    print(f"{'=' * 80}")
    
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        f1_scores.append(f1_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_val, y_pred, zero_division=0))
    
    results = {
        'method': method_name,
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores),
        'threshold': threshold
    }
    
    print(f"F1-Score:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print(f"Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
    print(f"Recall:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
    print(f"Threshold: {threshold:.3f}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("🚀 RANDOM FOREST HYPERPARAMETER OPTIMIZATION - VERSIÓN MEJORADA")
    print("=" * 80)
    print("\n📋 Mejoras implementadas:")
    print("  ✅ Eliminado StandardScaler (RF no lo necesita)")
    print("  ✅ Métrica: fbeta_score (beta=2, prioriza recall)")
    print("  ✅ TimeSeriesSplit: 5 splits (antes 3)")
    print("  ✅ Feature engineering: lags, rolling, momentum")
    print("  ✅ Threshold optimization: 100 puntos (antes ~10)")
    print("  ✅ Optuna: 200 trials (antes 50)")
    print("  ✅ Espacio de búsqueda mejorado")
    
    # Cargar datos
    X, y, feature_cols = load_and_prepare_data()
    
    # ✅ Mejora: 5 splits en lugar de 3
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    # Baseline
    print("\n" + "=" * 80)
    print("📊 BASELINE MODEL")
    print("=" * 80)
    baseline_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    baseline_results = evaluate_model(baseline_model, X, y, cv, "Baseline")
    
    # Grid Search
    grid_model, grid_params = run_grid_search(X, y, cv)
    grid_results = evaluate_model(grid_model, X, y, cv, "Grid Search")
    
    # Random Search
    random_model, random_params = run_random_search(X, y, cv)
    random_results = evaluate_model(random_model, X, y, cv, "Random Search")
    
    # Optuna
    optuna_model, optuna_params = run_optuna_optimization(X, y, cv)
    optuna_results = evaluate_model(optuna_model, X, y, cv, "Optuna")
    
    # Optimización de threshold (con el mejor modelo)
    best_threshold, threshold_results = optimize_threshold(optuna_model, X, y, cv)
    optuna_threshold_results = evaluate_model(
        optuna_model, X, y, cv, "Optuna + Threshold", threshold=best_threshold
    )
    
    # Comparación final
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN FINAL")
    print("=" * 80)
    
    comparison_df = pd.DataFrame([
        baseline_results,
        grid_results,
        random_results,
        optuna_results,
        optuna_threshold_results
    ])
    
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('hyperparameter_optimization_improved_results.csv', index=False)
    print("\n💾 Resultados guardados: hyperparameter_optimization_improved_results.csv")
    
    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['f1_mean', 'precision_mean', 'recall_mean']
    titles = ['F1-Score', 'Precision', 'Recall']
    
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(comparison_df['method'], comparison_df[metric])
        ax.set_ylabel(title)
        ax.set_title(f'{title} por Método')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_optimization_improved_comparison.png', dpi=300)
    print("💾 Gráfico guardado: hyperparameter_optimization_improved_comparison.png")
    
    print("\n" + "=" * 80)
    print("✅ OPTIMIZACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\n🏆 Mejor método: {comparison_df.loc[comparison_df['f1_mean'].idxmax(), 'method']}")
    print(f"🏆 Mejor F1-Score: {comparison_df['f1_mean'].max():.4f}")

if __name__ == "__main__":
    main()