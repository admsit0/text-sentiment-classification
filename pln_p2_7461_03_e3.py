"""
Práctica 2 - Ejercicio 3: Creación de ficheros de entrenamiento, validación y test
Procesamiento de Lenguaje Natural
Universidad Autónoma de Madrid
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetCreator:
    """Crea datasets balanceados para clasificación de polaridad"""
    
    def __init__(self, positive_threshold=7, neutral_min=5, neutral_max=6, 
                 negative_threshold=4):
        """
        Inicializa el creador de datasets
        
        Args:
            positive_threshold (int): Rating mínimo para clase positiva (>=)
            neutral_min (int): Rating mínimo para clase neutra
            neutral_max (int): Rating máximo para clase neutra
            negative_threshold (int): Rating máximo para clase negativa (<=)
        """
        self.positive_threshold = positive_threshold
        self.neutral_min = neutral_min
        self.neutral_max = neutral_max
        self.negative_threshold = negative_threshold
        
    def rating_to_label(self, rating):
        """
        Convierte rating numérico a etiqueta de polaridad
        
        Args:
            rating (float): Rating numérico (1-10)
            
        Returns:
            str: Etiqueta ('positive', 'neutral', 'negative')
        """
        if rating is None:
            return None
        
        if rating >= self.positive_threshold:
            return 'positive'
        elif self.neutral_min <= rating <= self.neutral_max:
            return 'neutral'
        elif rating <= self.negative_threshold:
            return 'negative'
        else:
            return None  # Ratings que no encajan en ninguna categoría
    
    def label_to_numeric(self, label):
        """
        Convierte etiqueta a valor numérico
        
        Args:
            label (str): Etiqueta ('positive', 'neutral', 'negative')
            
        Returns:
            int: 0=negative, 1=neutral, 2=positive
        """
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        return label_map.get(label, -1)
    
    def analyze_distribution(self, ratings):
        """
        Analiza la distribución de ratings en el corpus
        
        Args:
            ratings (list): Lista de ratings
            
        Returns:
            dict: Estadísticas de distribución
        """
        ratings = [r for r in ratings if r is not None]
        
        stats = {
            'count': len(ratings),
            'mean': np.mean(ratings),
            'median': np.median(ratings),
            'std': np.std(ratings),
            'min': np.min(ratings),
            'max': np.max(ratings),
            'distribution': Counter(ratings)
        }
        
        return stats
    
    def balance_dataset(self, vectors, labels, method='undersample', random_state=42):
        """
        Balancea el dataset para tener igual número de ejemplos por clase
        
        Args:
            vectors (np.array): Matriz de vectores
            labels (np.array): Array de etiquetas
            method (str): Método de balanceo ('undersample' o 'oversample')
            random_state (int): Semilla aleatoria
            
        Returns:
            tuple: (vectors_balanced, labels_balanced)
        """
        np.random.seed(random_state)
        
        unique_labels = np.unique(labels)
        label_counts = Counter(labels)
        
        print(f"\nDistribución original:")
        for label in unique_labels:
            print(f"  {label}: {label_counts[label]} ejemplos")
        
        if method == 'undersample':
            # Submuestreo: reducir todas las clases al tamaño de la minoritaria
            min_count = min(label_counts.values())
            print(f"\nAplicando submuestreo a {min_count} ejemplos por clase...")
            
            balanced_indices = []
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                selected_indices = np.random.choice(label_indices, min_count, replace=False)
                balanced_indices.extend(selected_indices)
            
            balanced_indices = np.array(balanced_indices)
            np.random.shuffle(balanced_indices)
            
            return vectors[balanced_indices], labels[balanced_indices]
        
        elif method == 'oversample':
            # Sobremuestreo: aumentar todas las clases al tamaño de la mayoritaria
            max_count = max(label_counts.values())
            print(f"\nAplicando sobremuestreo a {max_count} ejemplos por clase...")
            
            balanced_indices = []
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                selected_indices = np.random.choice(label_indices, max_count, replace=True)
                balanced_indices.extend(selected_indices)
            
            balanced_indices = np.array(balanced_indices)
            np.random.shuffle(balanced_indices)
            
            return vectors[balanced_indices], labels[balanced_indices]
        
        else:
            raise ValueError(f"Método de balanceo '{method}' no reconocido")
    
    def create_train_val_test_split(self, vectors, labels, 
                                    test_size=0.2, val_size=0.1,
                                    stratify=True, random_state=42):
        """
        Divide el dataset en conjuntos de entrenamiento, validación y test
        
        Args:
            vectors (np.array): Matriz de vectores
            labels (np.array): Array de etiquetas
            test_size (float): Proporción para test
            val_size (float): Proporción para validación (del conjunto train+val)
            stratify (bool): Mantener proporción de clases
            random_state (int): Semilla aleatoria
            
        Returns:
            dict: Diccionario con splits
        """
        stratify_param = labels if stratify else None
        
        # Primero dividir en train+val y test
        X_temp, X_test, y_temp, y_test = train_test_split(
            vectors, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Si hay validación, dividir train+val
        if val_size > 0:
            # Calcular el tamaño de validación respecto al conjunto temporal
            val_size_adjusted = val_size / (1 - test_size)
            stratify_temp = y_temp if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_temp
            )
            
            splits = {
                'train': {'X': X_train, 'y': y_train},
                'val': {'X': X_val, 'y': y_val},
                'test': {'X': X_test, 'y': y_test}
            }
        else:
            splits = {
                'train': {'X': X_temp, 'y': y_temp},
                'test': {'X': X_test, 'y': y_test}
            }
        
        # Mostrar estadísticas
        print("\nDistribución en los conjuntos:")
        for split_name, split_data in splits.items():
            y = split_data['y']
            label_counts = Counter(y)
            print(f"\n{split_name.upper()}:")
            print(f"  Total: {len(y)} ejemplos")
            for label in sorted(label_counts.keys()):
                percentage = (label_counts[label] / len(y)) * 100
                print(f"  {label}: {label_counts[label]} ({percentage:.1f}%)")
        
        return splits
    
    def save_splits(self, splits, representation_name, output_dir):
        """
        Guarda los splits en archivos
        
        Args:
            splits (dict): Diccionario con splits
            representation_name (str): Nombre de la representación
            output_dir (str): Directorio de salida
        """
        rep_dir = os.path.join(output_dir, representation_name)
        os.makedirs(rep_dir, exist_ok=True)
        
        for split_name, split_data in splits.items():
            # Guardar vectores
            X_file = os.path.join(rep_dir, f'{split_name}_X.npy')
            np.save(X_file, split_data['X'])
            
            # Guardar etiquetas
            y_file = os.path.join(rep_dir, f'{split_name}_y.npy')
            np.save(y_file, split_data['y'])
            
            print(f"  ✓ {split_name}: {X_file}, {y_file}")
        
        return rep_dir


def visualize_distribution(ratings, labels, thresholds, output_file):
    """
    Visualiza la distribución de ratings y etiquetas
    
    Args:
        ratings (list): Lista de ratings
        labels (list): Lista de etiquetas
        thresholds (dict): Umbrales usados
        output_file (str): Archivo de salida para la gráfica
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Histograma de ratings
    axes[0].hist(ratings, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(thresholds['positive'], color='green', linestyle='--', 
                   label=f'Positivo ≥ {thresholds["positive"]}')
    axes[0].axvline(thresholds['neutral_min'], color='orange', linestyle='--',
                   label=f'Neutro [{thresholds["neutral_min"]}, {thresholds["neutral_max"]}]')
    axes[0].axvline(thresholds['neutral_max'], color='orange', linestyle='--')
    axes[0].axvline(thresholds['negative'], color='red', linestyle='--',
                   label=f'Negativo ≤ {thresholds["negative"]}')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Ratings')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Distribución de etiquetas
    label_counts = Counter(labels)
    labels_sorted = sorted(label_counts.keys())
    counts = [label_counts[label] for label in labels_sorted]
    colors = {'negative': 'red', 'neutral': 'orange', 'positive': 'green'}
    bar_colors = [colors.get(label, 'gray') for label in labels_sorted]
    
    axes[1].bar(labels_sorted, counts, color=bar_colors, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Etiqueta')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Clases')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Añadir valores encima de las barras
    for i, (label, count) in enumerate(zip(labels_sorted, counts)):
        axes[1].text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfica guardada: {output_file}")
    plt.close()


def process_dataset_creation(vectors_dir, output_dir,
                            positive_threshold=7, neutral_min=5, 
                            neutral_max=6, negative_threshold=4,
                            balance_method='undersample',
                            test_size=0.2, val_size=0.1,
                            random_state=42):
    """
    Procesa todos los vectores y crea datasets para entrenamiento/validación/test
    
    Args:
        vectors_dir (str): Directorio con vectores del Ejercicio 2
        output_dir (str): Directorio de salida
        positive_threshold (int): Umbral para clase positiva
        neutral_min (int): Umbral mínimo para clase neutra
        neutral_max (int): Umbral máximo para clase neutra
        negative_threshold (int): Umbral máximo para clase negativa
        balance_method (str): Método de balanceo ('undersample' o 'oversample')
        test_size (float): Proporción para test
        val_size (float): Proporción para validación
        random_state (int): Semilla aleatoria
    """
    print("="*60)
    print("CREACIÓN DE DATASETS DE ENTRENAMIENTO, VALIDACIÓN Y TEST")
    print("="*60)
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar metadata
    metadata_file = os.path.join(vectors_dir, 'metadata.json')
    print(f"\nCargando metadata desde {metadata_file}...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    ratings = metadata['ratings']
    num_reviews = metadata['num_reviews']
    print(f"Total de reseñas: {num_reviews}")
    
    # Inicializar creador de datasets
    creator = DatasetCreator(
        positive_threshold=positive_threshold,
        neutral_min=neutral_min,
        neutral_max=neutral_max,
        negative_threshold=negative_threshold
    )
    
    # 1. ANALIZAR DISTRIBUCIÓN DE RATINGS
    print("\n" + "="*60)
    print("1. ANÁLISIS DE DISTRIBUCIÓN DE RATINGS")
    print("="*60)
    
    stats = creator.analyze_distribution(ratings)
    print(f"\nEstadísticas de ratings:")
    print(f"  Cantidad: {stats['count']}")
    print(f"  Media: {stats['mean']:.2f}")
    print(f"  Mediana: {stats['median']:.2f}")
    print(f"  Desviación estándar: {stats['std']:.2f}")
    print(f"  Rango: [{stats['min']:.1f}, {stats['max']:.1f}]")
    
    print(f"\nDistribución de ratings:")
    for rating in sorted(stats['distribution'].keys()):
        count = stats['distribution'][rating]
        percentage = (count / stats['count']) * 100
        print(f"  Rating {rating}: {count} ({percentage:.1f}%)")
    
    # 2. ETIQUETAR CORPUS
    print("\n" + "="*60)
    print("2. ETIQUETADO DEL CORPUS")
    print("="*60)
    
    print(f"\nUmbrales definidos:")
    print(f"  Positiva: rating ≥ {positive_threshold}")
    print(f"  Neutra: {neutral_min} ≤ rating ≤ {neutral_max}")
    print(f"  Negativa: rating ≤ {negative_threshold}")
    
    labels = [creator.rating_to_label(r) for r in ratings]
    labels_valid = [l for l in labels if l is not None]
    
    label_counts = Counter(labels_valid)
    print(f"\nDistribución de etiquetas:")
    total_valid = len(labels_valid)
    for label in ['positive', 'neutral', 'negative']:
        count = label_counts[label]
        percentage = (count / total_valid) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    discarded = labels.count(None)
    if discarded > 0:
        print(f"\n⚠ Reseñas descartadas (sin etiqueta válida): {discarded}")
    
    # Guardar configuración de etiquetado
    labeling_config = {
        'positive_threshold': positive_threshold,
        'neutral_min': neutral_min,
        'neutral_max': neutral_max,
        'negative_threshold': negative_threshold,
        'distribution': {
            'positive': label_counts['positive'],
            'neutral': label_counts['neutral'],
            'negative': label_counts['negative'],
            'discarded': discarded
        }
    }
    
    config_file = os.path.join(output_dir, 'labeling_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(labeling_config, f, indent=2)
    print(f"\n✓ Configuración guardada: {config_file}")
    
    # Visualizar distribución
    thresholds = {
        'positive': positive_threshold,
        'neutral_min': neutral_min,
        'neutral_max': neutral_max,
        'negative': negative_threshold
    }
    viz_file = os.path.join(output_dir, 'distribution.png')
    visualize_distribution(ratings, labels_valid, thresholds, viz_file)
    
    # 3. PROCESAR CADA REPRESENTACIÓN VECTORIAL
    print("\n" + "="*60)
    print("3. CREACIÓN DE SPLITS PARA CADA REPRESENTACIÓN")
    print("="*60)
    
    # Listar archivos de vectores
    vector_files = [f for f in os.listdir(vectors_dir) if f.startswith('vectors_') and f.endswith('.npy')]
    print(f"\nRepresentaciones encontradas: {len(vector_files)}")
    
    for vector_file in sorted(vector_files):
        representation_name = vector_file.replace('vectors_', '').replace('.npy', '')
        print(f"\n{'='*60}")
        print(f"Procesando: {representation_name}")
        print(f"{'='*60}")
        
        # Cargar vectores
        vector_path = os.path.join(vectors_dir, vector_file)
        vectors = np.load(vector_path)
        print(f"Vectores cargados: {vectors.shape}")
        
        # Filtrar vectores con etiquetas válidas
        valid_indices = [i for i, l in enumerate(labels) if l is not None]
        vectors_valid = vectors[valid_indices]
        labels_array = np.array([labels[i] for i in valid_indices])
        
        print(f"Vectores válidos: {vectors_valid.shape}")
        
        # Balancear dataset
        vectors_balanced, labels_balanced = creator.balance_dataset(
            vectors_valid, labels_array, method=balance_method, random_state=random_state
        )
        
        print(f"Vectores balanceados: {vectors_balanced.shape}")
        
        # Crear splits
        splits = creator.create_train_val_test_split(
            vectors_balanced, labels_balanced,
            test_size=test_size,
            val_size=val_size,
            stratify=True,
            random_state=random_state
        )
        
        # Guardar splits
        print(f"\nGuardando archivos en: {output_dir}/{representation_name}/")
        creator.save_splits(splits, representation_name, output_dir)
    
    # 4. RESUMEN FINAL
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    print(f"\nRepresentaciones procesadas: {len(vector_files)}")
    print(f"Directorio de salida: {output_dir}")
    print(f"\nEstructura de directorios creada:")
    for vector_file in sorted(vector_files):
        representation_name = vector_file.replace('vectors_', '').replace('.npy', '')
        print(f"  {representation_name}/")
        print(f"    ├── train_X.npy, train_y.npy")
        if val_size > 0:
            print(f"    ├── val_X.npy, val_y.npy")
        print(f"    └── test_X.npy, test_y.npy")
    
    print("\nArchivos adicionales:")
    print(f"  ✓ labeling_config.json (configuración de etiquetado)")
    print(f"  ✓ distribution.png (visualización de distribución)")
    
    print("\n" + "="*60)
    print("¡CREACIÓN DE DATASETS COMPLETADA!")
    print("="*60)


if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='Crea datasets de entrenamiento, validación y test'
    )
    parser.add_argument(
        '--vectors-dir',
        type=str,
        default='vectores',
        help='Directorio con vectores del Ejercicio 2'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets',
        help='Directorio de salida para datasets'
    )
    parser.add_argument(
        '--positive-threshold',
        type=float,
        default=7,
        help='Rating mínimo para clase positiva'
    )
    parser.add_argument(
        '--neutral-min',
        type=float,
        default=5,
        help='Rating mínimo para clase neutra'
    )
    parser.add_argument(
        '--neutral-max',
        type=float,
        default=6,
        help='Rating máximo para clase neutra'
    )
    parser.add_argument(
        '--negative-threshold',
        type=float,
        default=4,
        help='Rating máximo para clase negativa'
    )
    parser.add_argument(
        '--balance-method',
        type=str,
        default='undersample',
        choices=['undersample', 'oversample'],
        help='Método de balanceo del dataset'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporción para conjunto de test (0.0-1.0)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Proporción para conjunto de validación (0.0-1.0)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Semilla aleatoria para reproducibilidad'
    )
    
    args = parser.parse_args()
    
    # Ejecutar procesamiento
    process_dataset_creation(
        vectors_dir=args.vectors_dir,
        output_dir=args.output_dir,
        positive_threshold=args.positive_threshold,
        neutral_min=args.neutral_min,
        neutral_max=args.neutral_max,
        negative_threshold=args.negative_threshold,
        balance_method=args.balance_method,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
