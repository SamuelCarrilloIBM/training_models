#!/usr/bin/env python3
"""
XGBoost Classifier Mejorado para Trading de NVIDIA
===================================================

Mejoras implementadas:
1. Más regularización (alpha, lambda)
2. Menos profundidad (max_depth reducido)
3. Más ruido en entrenamiento (subsample más bajo)
4. Features adicionales:
   - Volatilidad de mercado (rolling std, ATR, Bollinger Bands)
   - Contexto de tendencia (ADX, tendencia a largo plazo)
   - Features macro (si disponibles)
5. Clasificación de 3 clases:
   - Clase 0: Bajar fuerte (< -1%)
   - Clase 1: Lateral (-1% a +1%)
   - Clase 2: Subir fuerte (> +1%)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def add_volatility_features(df):
    """
    Añade features de volatilidad de mercado
    """
    print("📊 Añadiendo features de volatilidad...")
    
    # Volatilidad rolling (diferentes ventanas)
    df['volatility_5d'] = df['close'].pct_change().rolling(5).std()
    df['volatility_10d'] = df['close'].pct_change().rolling(10).std()
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
    
    # ATR (Average True Range) - volatilidad real
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(14).mean()
    
    # Bollinger Bands Width (medida de volatilidad)
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volatilidad relativa (comparada con media histórica)
    df['volatility_ratio'] = df['volatility_20d'] / df['volatility_20d'].rolling(60).mean()
    
    # Rango intradiario normalizado
    df['intraday_range'] = (df['high'] - df['low']) / df['close']
    df['intraday_range_ma'] = df['intraday_range'].rolling(10).mean()
    
    # Limpieza de columnas temporales
    df.drop(['high_low', 'high_close', 'low_close', 'true_range', 'bb_middle', 'bb_std'], axis=1, inplace=True)
    
    return df


def add_trend_context_features(df):
    """
    Añade features de contexto de tendencia
    """
    print("📈 Añadiendo features de contexto de tendencia...")
    
    # ADX (Average Directional Index) - fuerza de la tendencia
    # Primero calculamos +DM y -DM
    df['high_diff'] = df['high'] - df['high'].shift(1)
    df['low_diff'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
    df['minus_dm'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0)
    
    # Calculamos ATR si no existe
    if 'atr_14' not in df.columns:
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df.drop(['high_low', 'high_close', 'low_close', 'true_range'], axis=1, inplace=True)
    
    # Directional Indicators
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr_14'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr_14'])
    
    # ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(14).mean()
    
    # Tendencia a largo plazo (múltiples timeframes)
    df['trend_50d'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
    df['trend_100d'] = (df['close'] - df['close'].rolling(100).mean()) / df['close'].rolling(100).mean()
    df['trend_200d'] = (df['close'] - df['close'].rolling(200).mean()) / df['close'].rolling(200).mean()
    
    # Posición relativa en el rango de N días
    df['position_in_range_20d'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
    df['position_in_range_50d'] = (df['close'] - df['close'].rolling(50).min()) / (df['close'].rolling(50).max() - df['close'].rolling(50).min())
    
    # Momentum de la tendencia
    df['trend_momentum'] = df['trend_50d'] - df['trend_50d'].shift(5)
    
    # Cruce de medias móviles (señal de cambio de tendencia)
    df['ma_cross_signal'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
    df['ma_cross_strength'] = abs(df['sma_20'] - df['sma_50']) / df['sma_50']
    
    # Limpieza de columnas temporales
    df.drop(['high_diff', 'low_diff', 'plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx'], axis=1, inplace=True)
    
    return df


def add_macro_features(df):
    """
    Añade features macro (basadas en datos disponibles)
    """
    print("🌍 Añadiendo features macro...")
    
    # Día de la semana (efecto calendario)
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Mes del año (estacionalidad)
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    
    # Días hasta fin de mes/trimestre (efecto window dressing)
    df['days_in_month'] = pd.to_datetime(df['date']).dt.days_in_month
    df['day_of_month'] = pd.to_datetime(df['date']).dt.day
    df['days_to_month_end'] = df['days_in_month'] - df['day_of_month']
    
    # Volatilidad del mercado (usando volumen como proxy)
    df['volume_volatility'] = df['volume'].pct_change().rolling(20).std()
    df['volume_trend'] = df['volume'].rolling(20).mean() / df['volume'].rolling(60).mean()
    
    # Momentum del mercado (acumulación de retornos)
    df['cumulative_return_5d'] = df['close'].pct_change(5)
    df['cumulative_return_10d'] = df['close'].pct_change(10)
    df['cumulative_return_20d'] = df['close'].pct_change(20)
    
    # Limpieza
    df.drop(['days_in_month', 'day_of_month'], axis=1, inplace=True)
    
    return df


def create_three_class_target(df, threshold_strong=0.01):
    """
    Crea target de 3 clases:
    - Clase 0: Bajar fuerte (< -threshold)
    - Clase 1: Lateral (-threshold a +threshold)
    - Clase 2: Subir fuerte (> +threshold)
    
    threshold_strong: umbral para considerar movimiento fuerte (default 1%)
    """
    print(f"🎯 Creando target de 3 clases (umbral: ±{threshold_strong*100:.1f}%)...")
    
    # Calcular retorno del día siguiente
    df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1
    
    # Crear clases
    conditions = [
        df['next_day_return'] < -threshold_strong,  # Bajar fuerte
        (df['next_day_return'] >= -threshold_strong) & (df['next_day_return'] <= threshold_strong),  # Lateral
        df['next_day_return'] > threshold_strong  # Subir fuerte
    ]
    choices = [0, 1, 2]
    
    df['target_3class'] = np.select(conditions, choices, default=1)
    
    # Estadísticas de distribución
    class_counts = df['target_3class'].value_counts().sort_index()
    print("\n📊 Distribución de clases:")
    print(f"  Clase 0 (Bajar fuerte): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Clase 1 (Lateral):      {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Clase 2 (Subir fuerte): {class_counts.get(2, 0)} ({class_counts.get(2, 0)/len(df)*100:.1f}%)")
    
    return df


def prepare_features(df):
    """
    Prepara el dataset con todas las features mejoradas
    """
    print("\n🔧 Preparando features mejoradas...")
    
    # Añadir nuevas features
    df = add_volatility_features(df)
    df = add_trend_context_features(df)
    df = add_macro_features(df)
    
    # Crear target de 3 clases
    df = create_three_class_target(df, threshold_strong=0.01)
    
    # Eliminar filas con NaN
    initial_rows = len(df)
    df = df.dropna()
    print(f"✅ Filas eliminadas por NaN: {initial_rows - len(df)}")
    print(f"✅ Filas finales: {len(df)}")
    
    return df


def train_improved_xgboost(X_train, y_train, X_val, y_val):
    """
    Entrena XGBoost con hiperparámetros mejorados:
    - Más regularización (alpha, lambda)
    - Menos profundidad (max_depth)
    - Más ruido (subsample, colsample_bytree)
    """
    print("\n🚀 Entrenando XGBoost mejorado...")
    
    # Hiperparámetros mejorados
    params = {
        # Parámetros de regularización (AUMENTADOS)
        'alpha': 10,  # L1 regularization (antes: 0)
        'lambda': 10,  # L2 regularization (antes: 1)
        'gamma': 5,  # Minimum loss reduction (antes: 0)
        
        # Parámetros de profundidad (REDUCIDOS)
        'max_depth': 3,  # Profundidad máxima (antes: 6)
        'min_child_weight': 5,  # Peso mínimo en nodo hijo (antes: 1)
        
        # Parámetros de ruido (AUMENTADOS)
        'subsample': 0.6,  # Fracción de muestras por árbol (antes: 0.8)
        'colsample_bytree': 0.6,  # Fracción de features por árbol (antes: 0.8)
        'colsample_bylevel': 0.6,  # Fracción de features por nivel (nuevo)
        
        # Parámetros de aprendizaje
        'learning_rate': 0.01,  # Learning rate bajo para mejor generalización
        'n_estimators': 500,  # Más árboles con learning rate bajo
        
        # Otros parámetros
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 50
    }
    
    print("\n📋 Hiperparámetros:")
    for key, value in params.items():
        if key not in ['objective', 'eval_metric', 'tree_method', 'random_state']:
            print(f"  {key}: {value}")
    
    # Entrenar modelo
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evalúa el modelo y genera reportes detallados
    """
    print("\n📊 Evaluando modelo...")
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Métricas generales
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {accuracy:.4f}")
    
    # Reporte de clasificación
    print("\n📋 Reporte de clasificación:")
    class_names = ['Bajar fuerte', 'Lateral', 'Subir fuerte']
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Matriz de confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Clase Real')
    axes[0, 0].set_xlabel('Clase Predicha')
    
    # 2. Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    axes[0, 1].barh(range(len(importance_df)), importance_df['importance'])
    axes[0, 1].set_yticks(range(len(importance_df)))
    axes[0, 1].set_yticklabels(importance_df['feature'])
    axes[0, 1].set_xlabel('Importancia')
    axes[0, 1].set_title('Top 20 Features Más Importantes', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    
    # 3. Distribución de probabilidades por clase
    for i, class_name in enumerate(class_names):
        axes[1, 0].hist(y_pred_proba[:, i], bins=50, alpha=0.5, label=class_name)
    axes[1, 0].set_xlabel('Probabilidad')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Probabilidades Predichas', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    
    # 4. Métricas por clase
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    axes[1, 1].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[1, 1].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[1, 1].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    axes[1, 1].set_xlabel('Clase')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Métricas por Clase', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('xgboost_improved_results.png', dpi=300, bbox_inches='tight')
    print("\n💾 Resultados guardados en: xgboost_improved_results.png")
    
    # Guardar feature importance
    importance_df.to_csv('xgboost_improved_feature_importance.csv', index=False)
    print("💾 Feature importance guardado en: xgboost_improved_feature_importance.csv")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    """
    Función principal
    """
    print("=" * 80)
    print("🚀 XGBoost Classifier Mejorado para Trading de NVIDIA")
    print("=" * 80)
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    df = pd.read_csv('data/dataset_nvda_lstm.csv')
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Preparar features
    df = prepare_features(df)
    
    # Separar features y target
    feature_cols = [col for col in df.columns if col not in ['date', 'target_3class', 'next_day_return']]
    X = df[feature_cols]
    y = df['target_3class']
    
    print(f"\n📊 Features finales: {len(feature_cols)}")
    print(f"📊 Muestras: {len(X)}")
    
    # Split temporal (80% train, 10% val, 10% test)
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\n📊 Split de datos:")
    print(f"  Train: {len(X_train)} muestras")
    print(f"  Val:   {len(X_val)} muestras")
    print(f"  Test:  {len(X_test)} muestras")
    
    # Normalizar features
    print("\n🔄 Normalizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = train_improved_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluar modelo
    results = evaluate_model(model, X_test_scaled, y_test, feature_cols)
    
    # Guardar modelo
    model.save_model('xgboost_improved_model.json')
    print("\n💾 Modelo guardado en: xgboost_improved_model.json")
    
    print("\n" + "=" * 80)
    print("✅ Entrenamiento completado exitosamente!")
    print("=" * 80)
    
    return model, results


if __name__ == "__main__":
    model, results = main()