"""
Práctica 2 - Ejercicio 4: Construcción de modelos de clasificación (PARALELIZADO)
Procesamiento de Lenguaje Natural
Universidad Autónoma de Madrid
"""

import json
import numpy as np
import pickle
import os
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost no está instalado. Instalar con: pip install xgboost")


class ModelTrainer:
    """Entrena y ajusta modelos de clasificación para análisis de sentimiento"""
    
    def __init__(self, random_state=42, n_jobs=-1):
        """
        Inicializa el entrenador de modelos
        
        Args:
            random_state (int): Semilla aleatoria
            n_jobs (int): Número de jobs paralelos (-1 = todos los cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trained_models = {}
        
    def get_model_configs(self):
        """
        Define las configuraciones de modelos y sus hiperparámetros
        
        Returns:
            dict: Diccionario con configuraciones de modelos
        """
        configs = {
            'multinomial_nb': {
                'name': 'Multinomial Naive Bayes',
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                    'fit_prior': [True, False]
                },
                'search_type': 'grid',
                'requires_positive': True  # Requiere valores positivos
            },
            'logistic_regression': {
                'name': 'Logistic Regression',
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=2000,  # Aumentado para evitar warnings
                    n_jobs=1,  # 1 por modelo para no sobrecargar con paralelización
                    solver='saga',  # Solver más robusto
                    tol=1e-4  # Tolerancia para convergencia
                ),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['saga']  # Solo saga (soporta L1 y L2)
                },
                'search_type': 'grid',
                'requires_positive': False
            },
            'svm_linear': {
                'name': 'SVM (Linear)',
                'model': SVC(
                    kernel='linear',
                    random_state=self.random_state,
                    probability=True,
                    cache_size=500,  # Aumentar caché para velocidad
                    max_iter=2000
                ),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'class_weight': [None, 'balanced']
                },
                'search_type': 'grid',
                'requires_positive': False
            },
            'svm_rbf': {
                'name': 'SVM (RBF)',
                'model': SVC(
                    kernel='rbf',
                    random_state=self.random_state,
                    probability=True,
                    cache_size=500,
                    max_iter=2000
                ),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'class_weight': [None, 'balanced']
                },
                'search_type': 'random',  # Demasiadas combinaciones
                'n_iter': 20,
                'requires_positive': False
            },
            'random_forest': {
                'name': 'Random Forest',
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=1,  # 1 por modelo para paralelización externa
                    warm_start=False
                ),
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': [None, 'balanced']
                },
                'search_type': 'random',
                'n_iter': 30,
                'requires_positive': False
            }
        }
        
        # Añadir XGBoost si está disponible
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'name': 'XGBoost',
                'model': XGBClassifier(
                    random_state=self.random_state,
                    n_jobs=1,  # 1 por modelo
                    eval_metric='mlogloss',
                    verbosity=0  # Silenciar warnings
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5]
                },
                'search_type': 'random',
                'n_iter': 30,
                'requires_positive': False
            }
        
        return configs
    
    def ensure_positive_values(self, X):
        """
        Asegura que todos los valores son positivos (para Multinomial NB)
        
        Args:
            X (np.array): Matriz de características
            
        Returns:
            tuple: (X_scaled, scaler)
        """
        # Si hay valores negativos, aplicar MinMaxScaler
        if np.any(X < 0):
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            return X_scaled, scaler
        return X, None
    
    def train_model(self, model_name, X_train, y_train, X_val=None, y_val=None,
                   use_validation=True, verbose=True):
        """
        Entrena un modelo con búsqueda de hiperparámetros
        
        Args:
            model_name (str): Nombre del modelo
            X_train (np.array): Características de entrenamiento
            y_train (np.array): Etiquetas de entrenamiento
            X_val (np.array): Características de validación (opcional)
            y_val (np.array): Etiquetas de validación (opcional)
            use_validation (bool): Usar conjunto de validación si está disponible
            verbose (bool): Mostrar información detallada
            
        Returns:
            dict: Resultados del entrenamiento
        """
        configs = self.get_model_configs()
        
        if model_name not in configs:
            raise ValueError(f"Modelo '{model_name}' no reconocido")
        
        config = configs[model_name]
        if verbose:
            print(f"\n{'='*60}")
            print(f"Entrenando: {config['name']}")
            print(f"{'='*60}")
        
        # Verificar si el modelo requiere valores positivos
        X_train_proc = X_train
        X_val_proc = X_val
        scaler = None
        
        if config.get('requires_positive', False):
            if verbose:
                print("  ℹ Modelo requiere valores positivos, aplicando MinMaxScaler...")
            X_train_proc, scaler = self.ensure_positive_values(X_train)
            if X_val is not None:
                X_val_proc = scaler.transform(X_val)
        
        # Configurar búsqueda de hiperparámetros
        scoring = 'f1_macro'  # F1-score macro (promedio entre clases)
        cv = 5  # 5-fold cross-validation
        
        # Determinar n_jobs para búsqueda (usar todos los cores disponibles)
        search_n_jobs = self.n_jobs if self.n_jobs != 1 else -1
        
        if config['search_type'] == 'grid':
            search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring=scoring,
                n_jobs=search_n_jobs,  # Paralelización en GridSearch
                verbose=0,  # Sin verbosidad interna
                return_train_score=True,
                error_score='raise',
                pre_dispatch='2*n_jobs'  # Optimización de memoria
            )
        else:  # random search
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=config.get('n_iter', 20),
                cv=cv,
                scoring=scoring,
                n_jobs=search_n_jobs,
                verbose=0,
                random_state=self.random_state,
                return_train_score=True,
                error_score='raise',
                pre_dispatch='2*n_jobs'
            )
        
        # Entrenar
        if verbose:
            print(f"  Búsqueda de hiperparámetros ({config['search_type']})...")
            print(f"  Cross-validation: {cv} folds")
            print(f"  Métrica: {scoring}")
        
        search.fit(X_train_proc, y_train)
        
        if verbose:
            print(f"\n  ✓ Entrenamiento completado")
            print(f"  Mejor score (CV): {search.best_score_:.4f}")
            print(f"  Mejores parámetros:")
            for param, value in search.best_params_.items():
                print(f"    - {param}: {value}")
        
        # Evaluar en validación si está disponible
        val_score = None
        if use_validation and X_val is not None and y_val is not None:
            val_score = search.best_estimator_.score(X_val_proc, y_val)
            if verbose:
                print(f"  Score en validación: {val_score:.4f}")
        
        # Guardar resultados
        results = {
            'model_name': model_name,
            'display_name': config['name'],
            'best_estimator': search.best_estimator_,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'cv_results': search.cv_results_,
            'val_score': val_score,
            'scaler': scaler  # Guardar scaler si se usó
        }
        
        return results
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None,
                        models_to_train=None, use_validation=True, parallel=True):
        """
        Entrena todos los modelos especificados (CON PARALELIZACIÓN)
        
        Args:
            X_train (np.array): Características de entrenamiento
            y_train (np.array): Etiquetas de entrenamiento
            X_val (np.array): Características de validación (opcional)
            y_val (np.array): Etiquetas de validación (opcional)
            models_to_train (list): Lista de nombres de modelos (None = todos)
            use_validation (bool): Usar conjunto de validación
            parallel (bool): Usar paralelización
            
        Returns:
            dict: Resultados de todos los modelos
        """
        configs = self.get_model_configs()
        
        if models_to_train is None:
            models_to_train = list(configs.keys())
        
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO DE MODELOS")
        print(f"{'='*60}")
        print(f"Modelos a entrenar: {len(models_to_train)}")
        print(f"Datos de entrenamiento: {X_train.shape}")
        if use_validation and X_val is not None:
            print(f"Datos de validación: {X_val.shape}")
        print(f"Paralelización: {'ACTIVADA' if parallel else 'DESACTIVADA'}")
        
        if parallel:
            # ENTRENAMIENTO PARALELO: Cada modelo en un proceso separado
            print(f"\n⚡ Entrenando {len(models_to_train)} modelos EN PARALELO...")
            
            def train_single_model(model_name, idx, total):
                """Función auxiliar para entrenar un modelo"""
                print(f"\n[{idx}/{total}] Iniciando: {configs[model_name]['name']}")
                try:
                    result = self.train_model(
                        model_name, X_train, y_train, X_val, y_val, 
                        use_validation, verbose=False
                    )
                    print(f"✓ [{idx}/{total}] Completado: {configs[model_name]['name']} "
                          f"(F1-CV: {result['best_cv_score']:.4f})")
                    return model_name, result
                except Exception as e:
                    print(f"✗ [{idx}/{total}] Error en {model_name}: {str(e)}")
                    return model_name, {'error': str(e)}
            
            # Entrenar todos los modelos en paralelo
            results_list = Parallel(n_jobs=-1, backend='loky', verbose=0)(
                delayed(train_single_model)(model_name, i+1, len(models_to_train))
                for i, model_name in enumerate(models_to_train)
            )
            
            # Convertir lista a diccionario
            results = {model_name: result for model_name, result in results_list}
            
        else:
            # ENTRENAMIENTO SECUENCIAL: Un modelo tras otro
            results = {}
            for i, model_name in enumerate(models_to_train, 1):
                print(f"\n{'#'*60}")
                print(f"Modelo {i}/{len(models_to_train)}")
                print(f"{'#'*60}")
                
                try:
                    result = self.train_model(
                        model_name, X_train, y_train, X_val, y_val, use_validation
                    )
                    results[model_name] = result
                except Exception as e:
                    print(f"  ✗ Error entrenando {model_name}: {str(e)}")
                    results[model_name] = {'error': str(e)}
        
        return results


def process_representation(rep_dir, rep_name, output_dir, models_to_train=None,
                          use_validation=True, random_state=42, n_jobs=-1, 
                          parallel=True):
    """
    Procesa una representación: entrena todos los modelos
    
    Args:
        rep_dir (str): Directorio con los datos de la representación
        rep_name (str): Nombre de la representación
        output_dir (str): Directorio de salida
        models_to_train (list): Lista de modelos a entrenar
        use_validation (bool): Usar conjunto de validación
        random_state (int): Semilla aleatoria
        n_jobs (int): Número de jobs paralelos
        parallel (bool): Usar paralelización
        
    Returns:
        dict: Resultados de entrenamiento
    """
    print("\n" + "="*60)
    print(f"PROCESANDO REPRESENTACIÓN: {rep_name}")
    print("="*60)
    
    # Cargar datos
    print("\nCargando datos...")
    X_train = np.load(os.path.join(rep_dir, 'train_X.npy'))
    y_train = np.load(os.path.join(rep_dir, 'train_y.npy'))
    print(f"  Train: {X_train.shape}, {len(y_train)} etiquetas")
    
    X_val = None
    y_val = None
    if use_validation and os.path.exists(os.path.join(rep_dir, 'val_X.npy')):
        X_val = np.load(os.path.join(rep_dir, 'val_X.npy'))
        y_val = np.load(os.path.join(rep_dir, 'val_y.npy'))
        print(f"  Val: {X_val.shape}, {len(y_val)} etiquetas")
    
    # Entrenar modelos
    trainer = ModelTrainer(random_state=random_state, n_jobs=n_jobs)
    results = trainer.train_all_models(
        X_train, y_train, X_val, y_val, models_to_train, use_validation, parallel
    )
    
    # Crear directorio de salida
    rep_output_dir = os.path.join(output_dir, rep_name)
    os.makedirs(rep_output_dir, exist_ok=True)
    
    # Guardar modelos
    print("\n" + "="*60)
    print("GUARDANDO MODELOS")
    print("="*60)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"  ✗ {model_name}: Error durante entrenamiento")
            continue
        
        # Guardar modelo completo
        model_file = os.path.join(rep_output_dir, f'{model_name}.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(result, f)
        print(f"  ✓ {model_name}: {model_file}")
    
    # Guardar resumen
    summary = {
        'representation': rep_name,
        'train_shape': list(X_train.shape),
        'val_shape': list(X_val.shape) if X_val is not None else None,
        'models': {}
    }
    
    for model_name, result in results.items():
        if 'error' not in result:
            summary['models'][model_name] = {
                'display_name': result['display_name'],
                'best_params': result['best_params'],
                'best_cv_score': float(result['best_cv_score']),
                'val_score': float(result['val_score']) if result['val_score'] is not None else None
            }
    
    summary_file = os.path.join(rep_output_dir, 'training_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Resumen: {summary_file}")
    
    return results


def main(datasets_dir, output_dir, representations=None, models=None,
         use_validation=True, random_state=42, n_jobs=-1, parallel=True,
         parallel_reps=False):
    """
    Función principal que procesa todas las representaciones
    
    Args:
        datasets_dir (str): Directorio con datasets del Ejercicio 3
        output_dir (str): Directorio de salida para modelos
        representations (list): Lista de representaciones a procesar (None = todas)
        models (list): Lista de modelos a entrenar (None = todos)
        use_validation (bool): Usar conjunto de validación
        random_state (int): Semilla aleatoria
        n_jobs (int): Número de jobs paralelos
        parallel (bool): Paralelizar entrenamiento de modelos
        parallel_reps (bool): Paralelizar procesamiento de representaciones
    """
    print("="*60)
    print("ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN (PARALELIZADO)")
    print("="*60)
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Listar representaciones disponibles
    available_reps = [
        d for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d))
    ]
    
    if representations is None:
        representations = available_reps
    else:
        # Filtrar solo las que existen
        representations = [r for r in representations if r in available_reps]
    
    print(f"\nRepresentaciones a procesar: {len(representations)}")
    for rep in representations:
        print(f"  • {rep}")
    
    if models:
        print(f"\nModelos a entrenar: {len(models)}")
        for model in models:
            print(f"  • {model}")
    else:
        print("\nModelos a entrenar: Todos los disponibles")
    
    print(f"\n⚡ Paralelización de modelos: {'ACTIVADA' if parallel else 'DESACTIVADA'}")
    print(f"⚡ Paralelización de representaciones: {'ACTIVADA' if parallel_reps else 'DESACTIVADA'}")
    
    # Procesar cada representación
    all_results = {}
    
    if parallel_reps and len(representations) > 1:
        # PROCESAMIENTO PARALELO DE REPRESENTACIONES
        print("\n⚡⚡ Procesando TODAS las representaciones EN PARALELO...")
        
        def process_single_rep(rep_name, idx, total):
            """Función auxiliar para procesar una representación"""
            print(f"\n[REP {idx}/{total}] Iniciando: {rep_name}")
            rep_dir = os.path.join(datasets_dir, rep_name)
            try:
                results = process_representation(
                    rep_dir, rep_name, output_dir, models,
                    use_validation, random_state, n_jobs, parallel
                )
                print(f"\n✓ [REP {idx}/{total}] Completado: {rep_name}")
                return rep_name, results
            except Exception as e:
                print(f"\n✗ [REP {idx}/{total}] Error en {rep_name}: {str(e)}")
                return rep_name, {'error': str(e)}
        
        results_list = Parallel(n_jobs=-1, backend='loky', verbose=0)(
            delayed(process_single_rep)(rep_name, i+1, len(representations))
            for i, rep_name in enumerate(representations)
        )
        
        all_results = {rep_name: result for rep_name, result in results_list}
        
    else:
        # PROCESAMIENTO SECUENCIAL DE REPRESENTACIONES
        for i, rep_name in enumerate(representations, 1):
            print("\n" + "#"*60)
            print(f"REPRESENTACIÓN {i}/{len(representations)}")
            print("#"*60)
            
            rep_dir = os.path.join(datasets_dir, rep_name)
            
            try:
                results = process_representation(
                    rep_dir, rep_name, output_dir, models,
                    use_validation, random_state, n_jobs, parallel
                )
                all_results[rep_name] = results
            except Exception as e:
                print(f"\n✗ Error procesando {rep_name}: {str(e)}")
                all_results[rep_name] = {'error': str(e)}
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    print(f"\nRepresentaciones procesadas: {len(all_results)}")
    print(f"Directorio de salida: {output_dir}")
    
    print("\n" + "="*60)
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)
    
    print("\nEstructura generada:")
    for rep_name in representations:
        print(f"  {rep_name}/")
        print(f"    ├── <modelo1>.pkl")
        print(f"    ├── <modelo2>.pkl")
        print(f"    └── training_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Entrena modelos de clasificación (PARALELIZADO)'
    )
    parser.add_argument(
        '--datasets-dir',
        type=str,
        default='datasets',
        help='Directorio con datasets del Ejercicio 3'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='modelos',
        help='Directorio de salida para modelos entrenados'
    )
    parser.add_argument(
        '--representations',
        nargs='+',
        default=None,
        help='Representaciones específicas a procesar (por defecto: todas)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['multinomial_nb', 'logistic_regression', 'svm_linear', 
                'svm_rbf', 'random_forest', 'xgboost'],
        default=None,
        help='Modelos específicos a entrenar (por defecto: todos)'
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='No usar conjunto de validación'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Semilla aleatoria'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Número de jobs paralelos (-1 = todos los cores)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Desactivar paralelización de modelos'
    )
    parser.add_argument(
        '--parallel-reps',
        action='store_true',
        help='Activar paralelización de representaciones (MÁXIMA VELOCIDAD)'
    )
    
    args = parser.parse_args()
    
    main(
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        representations=args.representations,
        models=args.models,
        use_validation=not args.no_validation,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        parallel=not args.no_parallel,
        parallel_reps=args.parallel_reps
    )