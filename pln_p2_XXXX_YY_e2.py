"""
Práctica 2 - Ejercicio 2: Generación de representaciones vectoriales
Procesamiento de Lenguaje Natural
Universidad Autónoma de Madrid
"""

import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import argparse

# Descargar recursos necesarios
def download_resources():
    """Descarga los recursos necesarios de NLTK"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        print("Advertencia: No se pudieron descargar algunos recursos de NLTK")


class VectorRepresentation:
    """Genera representaciones vectoriales de reseñas"""
    
    def __init__(self, use_stemming=False, use_lemmatization=True, 
                 remove_stopwords=True, max_features=5000):
        """
        Inicializa el generador de representaciones vectoriales
        
        Args:
            use_stemming (bool): Aplicar stemming
            use_lemmatization (bool): Aplicar lemmatization
            remove_stopwords (bool): Eliminar stopwords
            max_features (int): Número máximo de características para TF-IDF
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.max_features = max_features
        
        # Inicializar herramientas
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Vectorizadores (se entrenarán con el corpus)
        self.vectorizers = {}
        self.scalers = {}
        
    def preprocess_text(self, text):
        """
        Preprocesa el texto para la vectorización
        
        Args:
            text (str): Texto a preprocesar
            
        Returns:
            str: Texto preprocesado
        """
        # Tokenizar
        tokens = nltk.word_tokenize(text.lower())
        
        # Eliminar stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Aplicar stemming
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Aplicar lemmatization
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def create_ngram_representation(self, texts, ngram_range=(1, 1), 
                                   use_tfidf=True, vectorizer_name='unigram'):
        """
        Crea representación basada en n-gramas con TF-IDF o frecuencias
        
        Args:
            texts (list): Lista de textos
            ngram_range (tuple): Rango de n-gramas (min_n, max_n)
            use_tfidf (bool): Usar TF-IDF (True) o frecuencias (False)
            vectorizer_name (str): Nombre para guardar el vectorizador
            
        Returns:
            np.array: Matriz de representaciones vectoriales
        """
        print(f"Creando representación con n-gramas {ngram_range}...")
        
        # Preprocesar textos
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Crear vectorizador
        if use_tfidf:
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=self.max_features,
                min_df=2,  # Ignorar términos que aparecen en menos de 2 documentos
                max_df=0.95,  # Ignorar términos que aparecen en más del 95% de documentos
                sublinear_tf=True  # Aplicar escala logarítmica a TF
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=self.max_features,
                min_df=2,
                max_df=0.95
            )
        
        # Ajustar y transformar
        vectors = vectorizer.fit_transform(preprocessed_texts)
        
        # Guardar vectorizador
        self.vectorizers[vectorizer_name] = vectorizer
        
        print(f"  Dimensión del vector: {vectors.shape[1]}")
        print(f"  Vocabulario: {len(vectorizer.vocabulary_)} términos")
        
        return vectors.toarray()
    
    def extract_sentiment_features_vector(self, reviews_with_features):
        """
        Extrae características de sentimiento del Ejercicio 1 y las convierte en vector
        
        Args:
            reviews_with_features (list): Lista de reseñas con características extraídas
            
        Returns:
            np.array: Matriz de características de sentimiento
            list: Nombres de las características
        """
        print("Extrayendo características lingüísticas de sentimiento...")
        
        # Definir qué características numéricas extraer
        numeric_features = [
            # VADER
            'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu',
            
            # SentiWordNet
            'swn_pos_mean', 'swn_neg_mean', 'swn_pos_max', 'swn_neg_max',
            'swn_sentiment_word_count',
            
            # POS tags
            'adj_count', 'adv_count', 'verb_count',
            
            # Negaciones
            'negation_count', 'negated_sentiment_count', 'negative_pattern_count',
            
            # Modificadores
            'intensifier_count', 'diminisher_count', 'modifier_count',
            'intensifier_ratio',
            
            # Dominio
            'domain_mechanics_count', 'domain_components_count',
            'domain_rules_count', 'domain_experience_count',
            'domain_term_count', 'domain_term_ratio',
            'domain_sentiment_count',
            
            # Metadatos
            'token_count', 'sentence_count', 'avg_word_length'
        ]
        
        # Extraer valores para cada reseña
        feature_matrix = []
        for review in reviews_with_features:
            features = review['features']
            feature_vector = []
            
            for feat_name in numeric_features:
                value = features.get(feat_name, 0.0)
                # Asegurar que es numérico
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
                else:
                    feature_vector.append(0.0)
            
            feature_matrix.append(feature_vector)
        
        feature_matrix = np.array(feature_matrix)
        
        # Normalizar características (StandardScaler)
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Guardar scaler
        self.scalers['sentiment_features'] = scaler
        
        print(f"  Características extraídas: {len(numeric_features)}")
        print(f"  Dimensión del vector: {feature_matrix_scaled.shape[1]}")
        
        return feature_matrix_scaled, numeric_features
    
    def create_combined_representation(self, ngram_vectors, sentiment_vectors):
        """
        Combina representaciones de n-gramas y características de sentimiento
        
        Args:
            ngram_vectors (np.array): Vectores de n-gramas
            sentiment_vectors (np.array): Vectores de características de sentimiento
            
        Returns:
            np.array: Representación combinada
        """
        print("Combinando representaciones...")
        
        # Concatenar horizontalmente
        combined = np.hstack([ngram_vectors, sentiment_vectors])
        
        print(f"  Dimensión combinada: {combined.shape[1]}")
        print(f"    - N-gramas: {ngram_vectors.shape[1]}")
        print(f"    - Sentimiento: {sentiment_vectors.shape[1]}")
        
        return combined


def process_corpus_vectorization(input_file, output_dir, 
                                 ngram_configs=None,
                                 use_stemming=False,
                                 use_lemmatization=True,
                                 remove_stopwords=True,
                                 max_features=5000):
    """
    Procesa el corpus y genera todas las representaciones vectoriales
    
    Args:
        input_file (str): Archivo de entrada con características (del Ejercicio 1)
        output_dir (str): Directorio de salida para guardar representaciones
        ngram_configs (list): Lista de configuraciones de n-gramas
        use_stemming (bool): Aplicar stemming
        use_lemmatization (bool): Aplicar lemmatization
        remove_stopwords (bool): Eliminar stopwords
        max_features (int): Máximo de características para TF-IDF
    """
    import os
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuraciones por defecto de n-gramas
    if ngram_configs is None:
        ngram_configs = [
            {'range': (1, 1), 'name': 'unigram', 'use_tfidf': True},
            {'range': (1, 2), 'name': 'unigram_bigram', 'use_tfidf': True},
            {'range': (1, 3), 'name': 'unigram_bigram_trigram', 'use_tfidf': True}
        ]
    
    print("="*60)
    print("GENERACIÓN DE REPRESENTACIONES VECTORIALES")
    print("="*60)
    
    # Descargar recursos
    download_resources()
    
    # Cargar corpus con características
    print(f"\nCargando corpus desde {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    print(f"Corpus cargado: {len(corpus)} reseñas")
    
    # Extraer textos
    texts = [review['text'] for review in corpus]
    
    # Inicializar generador de representaciones
    vec_gen = VectorRepresentation(
        use_stemming=use_stemming,
        use_lemmatization=use_lemmatization,
        remove_stopwords=remove_stopwords,
        max_features=max_features
    )
    
    # Diccionario para almacenar todas las representaciones
    representations = {}
    
    # 1. REPRESENTACIONES BASADAS EN N-GRAMAS
    print("\n" + "="*60)
    print("1. REPRESENTACIONES BASADAS EN N-GRAMAS")
    print("="*60)
    
    for config in ngram_configs:
        ngram_range = config['range']
        name = config['name']
        use_tfidf = config.get('use_tfidf', True)
        
        print(f"\nGenerando representación: {name}")
        vectors = vec_gen.create_ngram_representation(
            texts, 
            ngram_range=ngram_range,
            use_tfidf=use_tfidf,
            vectorizer_name=name
        )
        
        representations[f'ngram_{name}'] = vectors
    
    # 2. REPRESENTACIÓN BASADA EN CARACTERÍSTICAS DE SENTIMIENTO
    print("\n" + "="*60)
    print("2. REPRESENTACIÓN BASADA EN CARACTERÍSTICAS DE SENTIMIENTO")
    print("="*60)
    
    sentiment_vectors, sentiment_feature_names = vec_gen.extract_sentiment_features_vector(corpus)
    representations['sentiment'] = sentiment_vectors
    
    # 3. REPRESENTACIONES COMBINADAS
    print("\n" + "="*60)
    print("3. REPRESENTACIONES COMBINADAS")
    print("="*60)
    
    for ngram_name in [config['name'] for config in ngram_configs]:
        print(f"\nCombinando n-gramas ({ngram_name}) + sentimiento")
        combined = vec_gen.create_combined_representation(
            representations[f'ngram_{ngram_name}'],
            sentiment_vectors
        )
        representations[f'combined_{ngram_name}_sentiment'] = combined
    
    # 4. GUARDAR REPRESENTACIONES
    print("\n" + "="*60)
    print("4. GUARDANDO REPRESENTACIONES")
    print("="*60)
    
    # Guardar cada representación
    for rep_name, rep_vectors in representations.items():
        output_file = os.path.join(output_dir, f'vectors_{rep_name}.npy')
        np.save(output_file, rep_vectors)
        print(f"  ✓ Guardado: {output_file} (shape: {rep_vectors.shape})")
    
    # Guardar vectorizadores y scalers
    vectorizers_file = os.path.join(output_dir, 'vectorizers.pkl')
    with open(vectorizers_file, 'wb') as f:
        pickle.dump(vec_gen.vectorizers, f)
    print(f"  ✓ Guardado: {vectorizers_file}")
    
    scalers_file = os.path.join(output_dir, 'scalers.pkl')
    with open(scalers_file, 'wb') as f:
        pickle.dump(vec_gen.scalers, f)
    print(f"  ✓ Guardado: {scalers_file}")
    
    # Guardar nombres de características de sentimiento
    sentiment_names_file = os.path.join(output_dir, 'sentiment_feature_names.json')
    with open(sentiment_names_file, 'w', encoding='utf-8') as f:
        json.dump(sentiment_feature_names, f, indent=2)
    print(f"  ✓ Guardado: {sentiment_names_file}")
    
    # Guardar IDs y ratings para referencia
    metadata = {
        'ids': [review.get('id', i) for i, review in enumerate(corpus)],
        'ratings': [review.get('rating') for review in corpus],
        'num_reviews': len(corpus)
    }
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Guardado: {metadata_file}")
    
    # Guardar configuración utilizada
    config_info = {
        'use_stemming': use_stemming,
        'use_lemmatization': use_lemmatization,
        'remove_stopwords': remove_stopwords,
        'max_features': max_features,
        'ngram_configs': ngram_configs,
        'representations': {
            name: {'shape': vectors.shape, 'dtype': str(vectors.dtype)}
            for name, vectors in representations.items()
        }
    }
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2)
    print(f"  ✓ Guardado: {config_file}")
    
    # 5. RESUMEN FINAL
    print("\n" + "="*60)
    print("RESUMEN DE REPRESENTACIONES GENERADAS")
    print("="*60)
    
    print(f"\nTotal de representaciones: {len(representations)}")
    print("\nDetalle:")
    for name, vectors in representations.items():
        print(f"  • {name:40s} → {vectors.shape}")
    
    print(f"\n✓ Todos los archivos guardados en: {output_dir}")
    print("\nArchivos generados:")
    print(f"  • {len(representations)} archivos .npy con vectores")
    print(f"  • vectorizers.pkl (vectorizadores TF-IDF/Count)")
    print(f"  • scalers.pkl (normalizadores)")
    print(f"  • sentiment_feature_names.json (nombres de características)")
    print(f"  • metadata.json (IDs y ratings)")
    print(f"  • config.json (configuración utilizada)")
    
    print("\n" + "="*60)
    print("¡VECTORIZACIÓN COMPLETADA!")
    print("="*60)


if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='Genera representaciones vectoriales de reseñas'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='corpus_con_features.json',
        help='Archivo de entrada con características (del Ejercicio 1)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='vectores',
        help='Directorio de salida para las representaciones'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Número máximo de características para TF-IDF'
    )
    parser.add_argument(
        '--no-lemmatization',
        action='store_true',
        help='Desactivar lemmatization'
    )
    parser.add_argument(
        '--use-stemming',
        action='store_true',
        help='Activar stemming'
    )
    parser.add_argument(
        '--keep-stopwords',
        action='store_true',
        help='No eliminar stopwords'
    )
    
    args = parser.parse_args()
    
    # Ejecutar procesamiento
    process_corpus_vectorization(
        input_file=args.input,
        output_dir=args.output_dir,
        use_stemming=args.use_stemming,
        use_lemmatization=not args.no_lemmatization,
        remove_stopwords=not args.keep_stopwords,
        max_features=args.max_features
    )
