"""
Práctica 2 - Ejercicio 4: Construcción de modelos de clasificación (PARALELIZADO)
Procesamiento de Lenguaje Natural
Universidad Autónoma de Madrid

-- REFACTORIZADO PARA FLUJO DE TRABAJO "TASK POOL" NO BLOQUEANTE --
-- VERSIÓN LIGERA CON GRIDSEARCH Y 4 MODELOS --
"""

import json
import numpy as np
import pickle
import os
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
# Eliminado: LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings('ignore')

# Eliminado: XGBoost (se asume que no se usa)


# Tamaño de la muestra para modelos extremadamente lentos (SVM RBF)
FAST_MODE_SAMPLE_SIZE = 5000


class ModelTrainer:
    """Entrena y ajusta modelos de clasificación para análisis de sentimiento"""
    
    def __init__(self, random_state=42, n_jobs=-1):
        """
        Inicializa el entrenador de modelos
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trained_models = {}
        
    def get_model_configs(self):
        """
        Define las configuraciones de modelos y sus hiperparámetros
        (Versión LIGERA con GridSearch)
        """
        configs = {
            'multinomial_nb': {
                'name': 'Multinomial Naive Bayes',
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.5, 1.0], 
                    'fit_prior': [True]
                },
                'search_type': 'grid', # CAMBIO: GridSearch
                # 'n_iter' se ignora
                'requires_positive': True,
                'requires_scaling': False,
                'fast_mode_subsample': False
            },
            # 'logistic_regression' ELIMINADO
            'svm_linear': {
                'name': 'SVM (LinearSVC)',
                'model': LinearSVC(
                    random_state=self.random_state,
                    max_iter=50,
                    dual='auto',
                    tol=1e-3
                ),
                'params': {
                    'C': [0.1, 1], 
                    'class_weight': [None, 'balanced']
                },
                'search_type': 'grid', # CAMBIO: GridSearch
                'requires_positive': False,
                'requires_scaling': True,
                'fast_mode_subsample': False
            },
            'svm_rbf': {
                'name': 'SVM (RBF)',
                'model': SVC(
                    kernel='rbf',
                    random_state=self.random_state,
                    probability=True,
                    cache_size=500
                ),
                'params': {
                    'C': [1, 10], 
                    'gamma': ['scale', 0.01], 
                },
                'search_type': 'grid', # CAMBIO: GridSearch
                'requires_positive': False,
                'requires_scaling': True,
                'fast_mode_subsample': True
            },
            'random_forest': {
                'name': 'Random Forest',
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=1,
                    warm_start=False
                ),
                'params': {
                    'n_estimators': [100], 
                    'max_depth': [10, None], 
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 4], 
                },
                'search_type': 'grid', # CAMBIO: GridSearch
                'requires_positive': False,
                'requires_scaling': False,
                'fast_mode_subsample': False
            }
            # 'xgboost' ELIMINADO
        }
        
        return configs
    
    # --- CORRECCIÓN DE MEMORIA: Forzar salida a float32 ---
    def ensure_positive_values(self, X):
        """Asegura que todos los valores son positivos (para Multinomial NB)"""
        if np.any(X < 0):
            scaler = MinMaxScaler()
            # Aplicar scaler y forzar de vuelta a float32
            X_scaled = scaler.fit_transform(X).astype(np.float32)
            return X_scaled, scaler
        # --- CORRECCIÓN DE MEMORIA: Asegurar float32 ---
        return X.astype(np.float32), None

    # --- CORRECCIÓN DE MEMORIA: Forzar salida a float32 ---
    def train_model(self, model_name, X_train, y_train, X_val=None, y_val=None,
                   use_validation=True, verbose=True, cv_n_jobs=-1,
                   cv_verbose=0): # <-- LOGGER AVANZADO
        """Entrena un modelo con búsqueda de hiperparámetros"""
        
        configs = self.get_model_configs()
        
        if model_name not in configs:
            raise ValueError(f"[LOGGER] Modelo '{model_name}' no reconocido")
        
        config = configs[model_name]
        if verbose:
            print(f"\n{'='*60}")
            print(f"[LOGGER] Entrenando: {config['name']}")
            print(f"{'='*60}")
        
        X_train_proc = X_train
        X_val_proc = X_val
        y_train_proc = y_train
        scaler = None
        
        if config.get('requires_positive', False):
            if verbose:
                print("  [LOGGER] ℹ Modelo requiere valores positivos, aplicando MinMaxScaler...")
            X_train_proc, scaler = self.ensure_positive_values(X_train)
            if X_val is not None and scaler is not None:
                # Aplicar scaler y forzar de vuelta a float32
                X_val_proc = scaler.transform(X_val).astype(np.float32)
                
        elif config.get('requires_scaling', False):
            if verbose:
                print("  [LOGGER] ℹ Modelo requiere escalado, aplicando StandardScaler...")
            scaler = StandardScaler()
            # Aplicar scaler y forzar de vuelta a float32
            X_train_proc = scaler.fit_transform(X_train).astype(np.float32)
            if X_val is not None:
                # Aplicar scaler y forzar de vuelta a float32
                X_val_proc = scaler.transform(X_val).astype(np.float32)
        
        if config.get('fast_mode_subsample', False) and len(y_train) > FAST_MODE_SAMPLE_SIZE:
            if verbose:
                print(f"  [LOGGER] ⚠ ¡FAST MODE! Submuestreando de {len(y_train)} a {FAST_MODE_SAMPLE_SIZE} para la búsqueda.")
            
            np.random.seed(self.random_state)
            indices = np.random.choice(len(y_train), FAST_MODE_SAMPLE_SIZE, replace=False)
            
            X_train_proc = X_train_proc[indices]
            y_train_proc = y_train[indices]
            
            if verbose:
                print(f"  [LOGGER] Datos de búsqueda: {X_train_proc.shape}")

        scoring = 'f1_macro'
        cv = 5
        search_n_jobs = cv_n_jobs
        
        # --- LOGGER AVANZADO: Se pasa 'cv_verbose' a GridSearchCV ---
        if config['search_type'] == 'grid':
            search = GridSearchCV(
                config['model'], config['params'], cv=cv, scoring=scoring,
                n_jobs=search_n_jobs, 
                verbose=cv_verbose, # <-- AQUÍ
                return_train_score=True,
                error_score='raise', pre_dispatch='2*n_jobs'
            )
        else:
            search = RandomizedSearchCV(
                config['model'], config['params'], n_iter=config.get('n_iter', 20),
                cv=cv, scoring=scoring, n_jobs=search_n_jobs, 
                verbose=cv_verbose, # <-- AQUÍ
                random_state=self.random_state, return_train_score=True,
                error_score='raise', pre_dispatch='2*n_jobs'
            )
        
        if verbose:
            print(f"  [LOGGER] Búsqueda de hiperparámetros ({config['search_type']})...")
            print(f"  [LOGGER] Datos de entrenamiento (búsqueda): {X_train_proc.shape}")
            print(f"  [LOGGER] Cross-validation: {cv} folds")
            print(f"  [LOGGER] Métrica: {scoring}")
            if search_n_jobs != 1:
                print(f"  [LOGGER] Paralelismo (CV): {search_n_jobs} jobs")
            if cv_verbose > 0:
                print(f"  [LOGGER AVANZADO ACTIVADO (Nivel {cv_verbose})] -> Verás el output de sklearn...")

        
        search.fit(X_train_proc, y_train_proc)
        
        if verbose:
            print(f"\n  [LOGGER] ✓ Entrenamiento completado")
            print(f"  [LOGGER] Mejor score (CV): {search.best_score_:.4f}")
            print(f"  [LOGGER] Mejores parámetros:")
            for param, value in search.best_params_.items():
                print(f"    - {param}: {value}")
        
        val_score = None
        if use_validation and X_val is not None and y_val is not None:
            val_score = search.best_estimator_.score(X_val_proc, y_val)
            if verbose:
                print(f"  [LOGGER] Score en validación: {val_score:.4f}")
        
        results = {
            'model_name': model_name,
            'display_name': config['name'],
            'best_estimator': search.best_estimator_,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'cv_results': search.cv_results_,
            'val_score': val_score,
            'scaler': scaler
        }
        
        return results

# -----------------------------------------------------------------
# NUEVA FUNCIÓN "WORKER" PARA EL POOL DE TAREAS
# -----------------------------------------------------------------
def process_task(task_info, datasets_dir, output_dir, use_validation, 
                 random_state, cv_verbose):
    """
    Función "worker" que ejecuta una única tarea (modelo + representación).
    Carga datos, entrena y guarda el .pkl.
    """
    
    # --- 1. Extraer info de la tarea ---
    task_id, total_tasks, rep_name, model_name = task_info
    start_time = time.time()
    
    print(f"\n[TAREA {task_id}/{total_tasks}] 🔥 INICIADA: Modelo '{model_name}' en '{rep_name}'")
    
    try:
        # --- 2. Cargar datos (E3) ---
        print(f"  [TAREA {task_id}/{total_tasks}] [LOGGER] Cargando datos para '{rep_name}'...")
        rep_datasets_dir = os.path.join(datasets_dir, rep_name)
        X_train = np.load(os.path.join(rep_datasets_dir, 'train_X.npy'))
        y_train = np.load(os.path.join(rep_datasets_dir, 'train_y.npy'))
        
        # Convertir etiquetas (para XGBoost, etc.)
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        if not np.issubdtype(y_train.dtype, np.number):
            y_train = np.vectorize(label_map.get)(y_train)

        X_val = None
        y_val = None
        if use_validation and os.path.exists(os.path.join(rep_datasets_dir, 'val_X.npy')):
            X_val = np.load(os.path.join(rep_datasets_dir, 'val_X.npy'))
            y_val = np.load(os.path.join(rep_datasets_dir, 'val_y.npy'))
            if not np.issubdtype(y_val.dtype, np.number):
                y_val = np.vectorize(label_map.get)(y_val)
        
        print(f"  [TAREA {task_id}/{total_tasks}] [LOGGER] Datos cargados: Train {X_train.shape}")

        # --- 3. Entrenar (E4) ---
        print(f"  [TAREA {task_id}/{total_tasks}] [LOGGER] Iniciando búsqueda de hiperparámetros...")
        trainer = ModelTrainer(random_state=random_state, n_jobs=1)
        result = trainer.train_model(
            model_name, X_train, y_train, X_val, y_val, 
            use_validation, 
            verbose=True, # <-- CAMBIO: activar logger propio para que imprima parámetros
            cv_n_jobs=1,   # <-- CRÍTICO: No anidar paralelismo
            cv_verbose=cv_verbose # <-- Pasar el logger avanzado
        )
        
        # --- 4. Guardar .pkl (E4) ---
        rep_output_dir = os.path.join(output_dir, rep_name)
        os.makedirs(rep_output_dir, exist_ok=True)
        
        # Este es el nombre de archivo "significativo"
        model_file = os.path.join(rep_output_dir, f'{model_name}.pkl')
        
        print(f"  [TAREA {task_id}/{total_tasks}] [LOGGER] Guardando modelo optimizado en {model_file}...")
        with open(model_file, 'wb') as f:
            pickle.dump(result, f)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ [TAREA {task_id}/{total_tasks}] COMPLETADA ({duration:.1f}s): Guardado '{model_file}'")
        
        # Devolver el resultado para el resumen final
        return (rep_name, model_name, result)
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"✗ [TAREA {task_id}/{total_tasks}] ERROR ({duration:.1f}s): Modelo '{model_name}' en '{rep_name}'")
        print(f"  Error: {e}")
        return (rep_name, model_name, {'error': str(e)})


# -----------------------------------------------------------------
# FUNCIÓN 'main' REESCRITA PARA EL POOL DE TAREAS
# -----------------------------------------------------------------
def main(datasets_dir, output_dir, representations=None, models=None,
         use_validation=True, random_state=42, n_jobs=-1,
         cv_verbose=0):
    """
    Función principal que procesa todas las tareas en un pool paralelo.
    """
    
    print("="*60)
    print("ENTRENAMIENTO DE MODELOS (E4) - MODO TASK POOL NO BLOQUEANTE")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Definir todas las representaciones ---
    available_reps = [
        d for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d))
    ]
    if representations is None:
        representations = available_reps
    else:
        representations = [r for r in representations if r in available_reps]
    
    # --- 2. Definir todos los modelos ---
    temp_trainer = ModelTrainer()
    available_models = list(temp_trainer.get_model_configs().keys())
    if models is None:
        models = available_models
    
    print(f"\n[LOGGER] Representaciones a procesar: {len(representations)}")
    for rep in representations: print(f"  • {rep}")
    
    print(f"\n[LOGGER] Modelos a entrenar: {len(models)}")
    for model in models: print(f"  • {model}")
    
    # --- 3. Crear la lista de tareas ---
    tasks = []
    task_id = 1
    for rep_name in representations:
        for model_name in models:
            tasks.append( (task_id, len(representations) * len(models), rep_name, model_name) )
            task_id += 1
            
    print(f"\n[LOGGER] Total de tareas a ejecutar: {len(tasks)}")
    
    # Determinar el número de workers (hilos)
    if n_jobs == -1:
        num_workers = os.cpu_count() or 1
    else:
        num_workers = n_jobs
    print(f"[LOGGER] ⚡ Iniciando pool de {num_workers} workers...")
    
    if cv_verbose > 0:
        print(f"📢 [LOGGER AVANZADO] Nivel de verbosidad de CV: {cv_verbose}")
        
    # --- 4. Ejecutar el pool de tareas paralelo ---
    start_time_total = time.time()
    
    results_list = Parallel(n_jobs=num_workers, backend='threading', verbose=0)(
        delayed(process_task)(
            task_info, datasets_dir, output_dir,
            use_validation, random_state, cv_verbose
        )
        for task_info in tasks
    )
    
    end_time_total = time.time()
    print(f"\n[LOGGER] ----------------------------------------------------")
    print(f"[LOGGER] 🏁 Pool de tareas completado en {(end_time_total - start_time_total) / 60:.2f} minutos.")
    print(f"[LOGGER] ----------------------------------------------------")

    # --- 5. Generar resúmenes (post-procesamiento) ---
    print("\n[LOGGER] Generando archivos 'training_summary.json'...")
    
    # Agrupar resultados por representación
    all_results = {}
    for rep_name, model_name, result in results_list:
        if rep_name not in all_results:
            all_results[rep_name] = {}
        all_results[rep_name][model_name] = result
    
    # Escribir un resumen por cada representación
    for rep_name, models_results in all_results.items():
        summary = {
            'representation': rep_name,
            'models': {}
        }
        for model_name, result in models_results.items():
            if 'error' not in result:
                summary['models'][model_name] = {
                    'display_name': result['display_name'],
                    'best_params': result['best_params'],
                    'best_cv_score': float(result['best_cv_score']),
                    'val_score': float(result['val_score']) if result['val_score'] is not None else None
                }
        
        summary_file = os.path.join(output_dir, rep_name, 'training_summary.json')
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            print(f"  ✓ Resumen guardado: {summary_file}")
        except Exception as e:
            print(f"  ✗ Error guardando resumen {summary_file}: {e}")

    print("\n" + "="*60)
    print("RESUMEN FINAL (E4)")
    print("="*60)
    print(f"\n[LOGGER] Representaciones procesadas: {len(all_results)}")
    print(f"[LOGGER] Directorio de salida: {output_dir}")
    print("\n" + "="*60)
    print("¡ENTRENAMIENTO (E4) COMPLETADO!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Entrena modelos de clasificación (PARALELIZADO) - E4'
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
    # --- CAMBIO: Actualizadas las choices ---
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['multinomial_nb', 'svm_linear', 
                'svm_rbf', 'random_forest'],
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
        help='Número de workers paralelos (hilos) (-1 = todos los cores)'
    )
    
    # --- LOGGER AVANZADO: Nuevo argumento ---
    parser.add_argument(
        '--cv-verbose',
        type=int,
        default=0,
        help='Nivel de verbosidad para GridSearchCV (0=silencioso, 3=avanzado)'
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
        cv_verbose=args.cv_verbose
    )
