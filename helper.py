"""
HELPER SCRIPT DE USO ÚNICO (INDEPENDIENTE DE LA PRÁCTICA)

Objetivo: "Ingeniería inversa" de los modelos .pkl generados por el E4.
          1. Identifica los modelos que se entrenaron con submuestreo
             (fast_mode_subsample=True).
          2. RE-ENTRENA solo esos modelos usando los parámetros óptimos
             encontrados, pero esta vez con el DATASET COMPLETO.
          3. Copia los modelos que ya estaban entrenados con el dataset
             completo.
          4. Guarda los 28 modelos finales en un nuevo directorio.
"""

import json
import numpy as np
import pickle
import os
import argparse
import time
import shutil
import warnings

# Importar TODAS las clases necesarias para que pickle.load() funcione
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Ignorar warnings
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN COPIADA DEL E4 ---
# (Solo para saber qué modelos re-entrenar)

# Tamaño de la muestra para modelos extremadamente lentos (SVM RBF)
FAST_MODE_SAMPLE_SIZE = 5000

def get_model_configs():
    """
    Define las configuraciones de modelos y sus hiperparámetros
    (Copiado del E4 para saber qué modelos tenían 'fast_mode_subsample')
    """
    configs = {
        'multinomial_nb': {
            'name': 'Multinomial Naive Bayes',
            'model': MultinomialNB(),
            'params': {}, 'search_type': 'grid',
            'requires_positive': True,
            'requires_scaling': False,
            'fast_mode_subsample': False # <-- No re-entrenar
        },
        'svm_linear': {
            'name': 'SVM (LinearSVC)',
            'model': LinearSVC(random_state=42, max_iter=50, dual='auto', tol=1e-3),
            'params': {}, 'search_type': 'grid',
            'requires_positive': False,
            'requires_scaling': True,
            'fast_mode_subsample': False # <-- No re-entrenar
        },
        'svm_rbf': {
            'name': 'SVM (RBF)',
            'model': SVC(kernel='rbf', random_state=42, probability=True, cache_size=500),
            'params': {}, 'search_type': 'grid',
            'requires_positive': False,
            'requires_scaling': True,
            'fast_mode_subsample': True # <-- ¡SÍ RE-ENTRENAR!
        },
        'random_forest': {
            'name': 'Random Forest',
            'model': RandomForestClassifier(random_state=42, n_jobs=1, warm_start=False),
            'params': {}, 'search_type': 'grid',
            'requires_positive': False,
            'requires_scaling': False,
            'fast_mode_subsample': False # <-- No re-entrenar
        }
    }
    return configs

# --- FIN DE LA CONFIGURACIÓN COPIADA ---


def retrain_optimal_models(modelos_dir, datasets_dir, output_dir, random_state=42):
    """
    Itera sobre todos los modelos, re-entrena los necesarios y 
    guarda la colección final.
    """
    
    print("="*80)
    print("INICIANDO SCRIPT DE RE-ENTRENAMIENTO ÓPTIMO")
    print("="*80)
    print(f"[LOGGER] Buscando modelos originales en: {modelos_dir}")
    print(f"[LOGGER] Usando datasets completos de: {datasets_dir}")
    print(f"[LOGGER] Directorio de salida: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    model_configs = get_model_configs()

    try:
        representations = sorted([d for d in os.listdir(modelos_dir) if os.path.isdir(os.path.join(modelos_dir, d))])
    except FileNotFoundError:
        print(f"\n[LOGGER] ✗ ERROR: No se encuentra el directorio de modelos: {modelos_dir}")
        return

    print(f"\n[LOGGER] Se encontraron {len(representations)} representaciones.")
    
    total_copied = 0
    total_retrained = 0
    
    # 1. Iterar por Representaciones (carpetas en 'modelos/')
    for rep_name in representations:
        print(f"\n" + "-"*80)
        print(f"REPRESENTACIÓN: {rep_name}")
        
        rep_model_dir = os.path.join(modelos_dir, rep_name)
        rep_dataset_dir = os.path.join(datasets_dir, rep_name)
        rep_output_dir = os.path.join(output_dir, rep_name)
        os.makedirs(rep_output_dir, exist_ok=True)

        # 2. Iterar por Modelos (.pkl)
        model_files = sorted([f for f in os.listdir(rep_model_dir) if f.endswith('.pkl')])
        
        if not model_files:
            print("  [LOGGER] ⚠ No se encontraron modelos .pkl en este directorio.")
            continue
            
        print(f"  [LOGGER] Se encontraron {len(model_files)} modelos .pkl. Procesando...")
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            original_pkl_path = os.path.join(rep_model_dir, model_file)
            new_pkl_path = os.path.join(rep_output_dir, model_file)
            
            print(f"\n    --------------------------------------------------")
            print(f"    MODELO: {model_name} (en {rep_name})")
            
            if model_name not in model_configs:
                print(f"    [LOGGER] 🟡 OMITIDO: El modelo '{model_name}' no está en la configuración.")
                continue

            config = model_configs[model_name]

            # --- 3. Decidir si copiar o re-entrenar ---
            
            # CASO 1: Modelo RÁPIDO. Ya se entrenó con datos completos. Solo copiar.
            if not config.get('fast_mode_subsample', False):
                print(f"    [LOGGER] ℹ No requiere re-entrenamiento (fast_mode_subsample=False).")
                try:
                    shutil.copy(original_pkl_path, new_pkl_path)
                    print(f"    [LOGGER] ✅ COPIADO a {new_pkl_path}")
                    total_copied += 1
                except Exception as e:
                    print(f"    [LOGGER] ✗ ERROR al copiar {original_pkl_path}: {e}")
            
            # CASO 2: Modelo LENTO (svm_rbf). Requiere re-entrenamiento.
            else:
                print(f"    [LOGGER] 🔥 RE-ENTRENAMIENTO REQUERIDO (fast_mode_subsample=True).")
                start_time = time.time()
                try:
                    # Cargar el .pkl original para obtener los hiperparámetros
                    print(f"      [LOGGER] Cargando {original_pkl_path} para leer params...")
                    with open(original_pkl_path, 'rb') as f:
                        model_info = pickle.load(f)
                    
                    best_params = model_info['best_params']
                    print(f"      [LOGGER] Parámetros óptimos encontrados: {best_params}")

                    # Cargar el dataset COMPLETO de entrenamiento
                    print(f"      [LOGGER] Cargando dataset COMPLETO desde {rep_dataset_dir}...")
                    X_train = np.load(os.path.join(rep_dataset_dir, 'train_X.npy'))
                    y_train = np.load(os.path.join(rep_dataset_dir, 'train_y.npy'))
                    
                    # Convertir etiquetas
                    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
                    if not np.issubdtype(y_train.dtype, np.number):
                        y_train = np.vectorize(label_map.get)(y_train)
                    
                    print(f"      [LOGGER] Datos cargados: Train {X_train.shape}")
                    
                    # Crear una nueva instancia del modelo base
                    # (Importante: debe tener el mismo random_state que el original)
                    base_model = config['model']
                    optimal_model = base_model.set_params(**best_params)
                    
                    # Aplicar el scaler (¡crear uno nuevo y ajustarlo a los datos COMPLETOS!)
                    X_train_proc = X_train
                    new_scaler = None
                    if config.get('requires_positive', False):
                        print("      [LOGGER] Ajustando NUEVO MinMaxScaler en datos completos...")
                        new_scaler = MinMaxScaler()
                        X_train_proc = new_scaler.fit_transform(X_train).astype(np.float32)
                    elif config.get('requires_scaling', False):
                        print("      [LOGGER] Ajustando NUEVO StandardScaler en datos completos...")
                        new_scaler = StandardScaler()
                        X_train_proc = new_scaler.fit_transform(X_train).astype(np.float32)

                    # --- RE-ENTRENAMIENTO ---
                    print(f"      [LOGGER] 🔥 RE-ENTRENANDO {model_name} en {X_train_proc.shape} datos...")
                    optimal_model.fit(X_train_proc, y_train)
                    duration = time.time() - start_time
                    print(f"      [LOGGER] ✅ Re-entrenamiento completado en {duration:.1f}s.")
                    
                    # Crear el nuevo diccionario para guardar (igual que en E4)
                    new_model_info = {
                        'model_name': model_info['model_name'],
                        'display_name': model_info['display_name'],
                        'best_estimator': optimal_model, # <-- El modelo NUEVO
                        'best_params': best_params,
                        'best_cv_score': model_info['best_cv_score'], # Mantenemos el score de la búsqueda
                        'cv_results': model_info['cv_results'],
                        'val_score': model_info['val_score'], # Mantenemos el score de validación
                        'scaler': new_scaler # <-- El scaler NUEVO
                    }
                    
                    # Guardar el nuevo .pkl
                    print(f"      [LOGGER] Guardando NUEVO modelo en {new_pkl_path}...")
                    with open(new_pkl_path, 'wb') as f:
                        pickle.dump(new_model_info, f)
                    
                    print(f"    [LOGGER] ✅ RE-ENTRENADO y GUARDADO en {new_pkl_path}")
                    total_retrained += 1

                except Exception as e:
                    print(f"    [LOGGER] ✗ ERROR durante el re-entrenamiento de {model_name}: {e}")

    print("\n" + "="*80)
    print("PROCESO DE OPTIMIZACIÓN COMPLETADO")
    print("="*80)
    print(f"  Modelos copiados (sin cambios): {total_copied}")
    print(f"  Modelos re-entrenados (óptimos): {total_retrained}")
    print(f"  Total de modelos en '{output_dir}': {total_copied + total_retrained}")
    print("\n¡Listo para ejecutar el Ejercicio 5 sobre el directorio 'OPTIMAL_MODELS'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Helper script para re-entrenar modelos óptimos con el dataset completo.'
    )
    parser.add_argument(
        '--modelos-dir',
        type=str,
        default='modelos',
        help='Directorio donde están los modelos entrenados (SALIDA del E4)'
    )
    parser.add_argument(
        '--datasets-dir',
        type=str,
        default='datasets',
        help='Directorio con los datasets (SALIDA del E3)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='OPTIMAL_MODELS',
        help='Directorio NUEVO donde se guardarán los modelos finales (óptimos)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Semilla aleatoria (debe coincidir con la usada en E4)'
    )
    
    args = parser.parse_args()
    
    retrain_optimal_models(
        modelos_dir=args.modelos_dir,
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        random_state=args.random_state
    )