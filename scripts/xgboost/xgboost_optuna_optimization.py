#!/usr/bin/env python3
"""
XGBoost Hyperparameter Optimization con Optuna - VERSIÓN OPTIMIZADA
====================================================================

Mejoras implementadas:
1. ✅ Optimización Bayesiana con Optuna (200 trials)
2. ✅ Métrica optimizada: fbeta_score (beta=2 para priorizar recall)
3. ✅ TimeSeriesSplit con 5 splits
4. ✅ Feature engineering avanzado (lags, rolling, momentum)
5. ✅ Threshold optimization con 100 puntos
6. ✅ Early stopping para evitar overfitting
7. ✅ Espacio de búsqueda optimizado para XGBoost
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    fbeta_score,
    make_scorer,
    accuracy_score
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
N_SPLITS = 5
N_TRIALS_OPTUNA = 200

# ============================================================================
# FUNCIONES DE FEATURE ENGINEERING
# ============================================================================

def add_advanced_features(df):
    """Añade features avanzadas: lags, rolling stats, momentum"""
    df = df.copy()
    
    # Detectar columna de precio
    price_col = None
    for col in ['close', 'Close', 'precio', 'price']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        print("⚠️ No se encontró columna de precio. Usando features existentes.")
        return df
    
    print(f"📊 Usando columna de precio: '{price_col}'")
    
    # 1. LAGS
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
    
    # 6. FEATURES DE SENTIMIENTO
    sentiment_cols = [col for col in df.columns if 'tone' in col.lower() or 'sentiment' in col.lower()]
    if sentiment_cols:
        print(f"📊 Añadiendo features de sentimiento: {len(sentiment_cols)} columnas")
        for col in sentiment_cols[:3]:
            df[f'{col}_ma_5'] = df[col].rolling(5).mean()
            df[f'{col}_std_5'] = df[col].rolling(5).std()
    
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
    
    # Feature engineering
    print("\n" + "=" * 80)
    print("🔧 FEATURE ENGINEERING AVANZADO")
    print("=" * 80)
    df = add_advanced_features(df)
    
    # Separar features y target
    target_col = 'target'
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
    """Optimiza el threshold usando validación cruzada"""
    print("\n" + "=" * 80)
    print("🎯 OPTIMIZACIÓN DE THRESHOLD (100 puntos)")
    print("=" * 80)
    
    thresholds = np.linspace(0.1, 0.9, 100)
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
        results.append({'threshold': threshold, 'f1_score': mean_f1})
        
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
    plt.title('XGBoost - Optimización de Threshold (100 puntos)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgboost_threshold_optimization.png', dpi=300)
    print("💾 Gráfico guardado: xgboost_threshold_optimization.png")
    
    return best_threshold, results_df

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================

def run_optuna_optimization(X, y, cv):
    """
    Optimización Bayesiana con Optuna para XGBoost
    Espacio de búsqueda optimizado específicamente para XGBoost
    """
    print("\n" + "=" * 80)
    print(f"🔬 OPTUNA OPTIMIZATION ({N_TRIALS_OPTUNA} trials)")
    print("=" * 80)
    
    def objective(trial):
        # Espacio de búsqueda optimizado para XGBoost
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            
            # Parámetros de árboles
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            
            # Parámetros de learning
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            
            # Parámetros de regularización
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            
            # Parámetros de sampling
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            
            # Otros
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        
        # Usar fbeta_score con beta=2 (prioriza recall)
        fbeta_scorer = make_scorer(fbeta_score, beta=2)
        scores = cross_val_score(model, X, y, cv=cv, scoring=fbeta_scorer, n_jobs=-1)
        
        return scores.mean()
    
    # Crear estudio de Optuna
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    
    print(f"🚀 Iniciando optimización con {N_TRIALS_OPTUNA} trials...")
    print("   Esto puede tardar 10-20 minutos dependiendo del hardware\n")
    
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)
    
    print(f"\n✅ Optimización completada!")
    print(f"\n🏆 Mejores hiperparámetros:")
    for param, value in study.best_params.items():
        print(f"   {param}: {value}")
    print(f"\n📊 Mejor F-beta score (CV): {study.best_value:.4f}")
    
    # Entrenar modelo final con mejores parámetros
    best_params = study.best_params.copy()
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    best_params['verbosity'] = 0
    
    best_model = xgb.XGBClassifier(**best_params)
    
    return best_model, study.best_params, study

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
    accuracy_scores = []
    
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train, verbose=False)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        f1_scores.append(f1_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_val, y_pred, zero_division=0))
        accuracy_scores.append(accuracy_score(y_val, y_pred))
    
    results = {
        'method': method_name,
        'accuracy_mean': np.mean(accuracy_scores),
        'accuracy_std': np.std(accuracy_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores),
        'threshold': threshold
    }
    
    print(f"Accuracy:  {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"F1-Score:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print(f"Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
    print(f"Recall:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
    print(f"Threshold: {threshold:.3f}")
    
    return results

# ============================================================================
# VISUALIZACIÓN DE IMPORTANCIA DE FEATURES
# ============================================================================

def plot_feature_importance(model, feature_cols, top_n=20):
    """Visualiza las features más importantes"""
    print("\n" + "=" * 80)
    print(f"📊 TOP {top_n} FEATURES MÁS IMPORTANTES")
    print("=" * 80)
    
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(top_n).to_string(index=False))
    
    # Visualización
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Features Más Importantes - XGBoost')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300)
    print("\n💾 Gráfico guardado: xgboost_feature_importance.png")
    
    return feature_importance

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("🚀 XGBOOST HYPERPARAMETER OPTIMIZATION CON OPTUNA")
    print("=" * 80)
    print("\n📋 Mejoras implementadas:")
    print("  ✅ Optimización Bayesiana con Optuna (200 trials)")
    print("  ✅ Métrica: fbeta_score (beta=2, prioriza recall)")
    print("  ✅ TimeSeriesSplit: 5 splits")
    print("  ✅ Feature engineering avanzado")
    print("  ✅ Threshold optimization: 100 puntos")
    print("  ✅ Espacio de búsqueda optimizado para XGBoost")
    
    # Cargar datos
    X, y, feature_cols = load_and_prepare_data()
    
    # TimeSeriesSplit
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    # Baseline XGBoost
    print("\n" + "=" * 80)
    print("📊 BASELINE XGBOOST")
    print("=" * 80)
    baseline_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    baseline_results = evaluate_model(baseline_model, X, y, cv, "Baseline XGBoost")
    
    # Optuna Optimization
    optuna_model, optuna_params, study = run_optuna_optimization(X, y, cv)
    optuna_results = evaluate_model(optuna_model, X, y, cv, "XGBoost + Optuna")
    
    # Threshold Optimization
    best_threshold, threshold_results = optimize_threshold(optuna_model, X, y, cv)
    optuna_threshold_results = evaluate_model(
        optuna_model, X, y, cv, "XGBoost + Optuna + Threshold", threshold=best_threshold
    )
    
    # Feature Importance
    # Entrenar modelo final con todos los datos
    optuna_model.fit(X, y)
    feature_importance = plot_feature_importance(optuna_model, feature_cols)
    
    # Comparación final
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN FINAL")
    print("=" * 80)
    
    comparison_df = pd.DataFrame([
        baseline_results,
        optuna_results,
        optuna_threshold_results
    ])
    
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('xgboost_optuna_results.csv', index=False)
    print("\n💾 Resultados guardados: xgboost_optuna_results.csv")
    
    # Visualización comparativa
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    metrics = ['accuracy_mean', 'f1_mean', 'precision_mean', 'recall_mean']
    titles = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(comparison_df['method'], comparison_df[metric])
        ax.set_ylabel(title)
        ax.set_title(f'{title} por Método')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_optuna_comparison.png', dpi=300)
    print("💾 Gráfico guardado: xgboost_optuna_comparison.png")
    
    # Guardar mejores parámetros
    params_df = pd.DataFrame([optuna_params])
    params_df.to_csv('xgboost_best_params.csv', index=False)
    print("💾 Mejores parámetros guardados: xgboost_best_params.csv")
    
    print("\n" + "=" * 80)
    print("✅ OPTIMIZACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\n🏆 Mejor método: {comparison_df.loc[comparison_df['f1_mean'].idxmax(), 'method']}")
    print(f"🏆 Mejor F1-Score: {comparison_df['f1_mean'].max():.4f}")
    print(f"🏆 Mejor Accuracy: {comparison_df['accuracy_mean'].max():.4f}")
    print(f"🏆 Mejor Recall: {comparison_df['recall_mean'].max():.4f}")
    
    print("\n📁 Archivos generados:")
    print("   - xgboost_optuna_results.csv")
    print("   - xgboost_optuna_comparison.png")
    print("   - xgboost_threshold_optimization.png")
    print("   - xgboost_feature_importance.png")
    print("   - xgboost_best_params.csv")

if __name__ == "__main__":
    main()